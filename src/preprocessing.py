"""Data loading, cleaning and feature engineering.

Implements the preprocessing pipeline described in the paper:

* median-fill ``CPI`` and ``Unemployment``
* ``MarkDown1-5`` negatives -> 0, missing -> 0, summed into ``TotalMarkDown``
* merge ``stores`` + ``features`` onto the sales rows
* derive ``Year`` / ``Month`` / ``Week`` from ``Date``
* z-score outlier removal on store/department weekly sales
* one-hot encode ``Store`` / ``Dept`` / ``Type``
* min-max scale the target
* Recursive Feature Elimination (RFE) for feature selection

It produces two views of the data:

``build_feature_matrix`` -> tabular (X, y) for Random Forest / ANN
``build_weekly_series``   -> a single weekly aggregated series for the
                             time-series models and the hybrids
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler

from . import config

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw() -> dict[str, pd.DataFrame]:
    """Read the four source CSVs. Raises a helpful error if they're missing."""
    missing = [p.name for p in (config.STORES_CSV, config.TRAIN_CSV,
                                 config.FEATURES_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset file(s): {missing}. "
            "See data/README.md for how to download them into ./data."
        )
    return {
        "stores": pd.read_csv(config.STORES_CSV),
        "train": pd.read_csv(config.TRAIN_CSV, parse_dates=["Date"]),
        "features": pd.read_csv(config.FEATURES_CSV, parse_dates=["Date"]),
    }


# ---------------------------------------------------------------------------
# Cleaning + merge + feature engineering
# ---------------------------------------------------------------------------
def clean_and_merge(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge the tables and apply the cleaning rules from the paper."""
    features = raw["features"].copy()

    # MarkDown: negatives -> 0, missing -> 0.
    for col in MARKDOWN_COLS:
        if col in features:
            features[col] = features[col].clip(lower=0).fillna(0)

    # Median-fill the macro-economic columns.
    for col in ("CPI", "Unemployment"):
        if col in features:
            features[col] = features[col].fillna(features[col].median())

    # Merge sales <- features (Store, Date) <- stores (Store).
    df = raw["train"].merge(
        features, on=["Store", "Date"], how="left", suffixes=("", "_feat")
    )
    # IsHoliday appears in both; keep the sales-table version.
    if "IsHoliday_feat" in df:
        df = df.drop(columns=["IsHoliday_feat"])
    df = df.merge(raw["stores"], on="Store", how="left")

    # Calendar features.
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    # Total markdown, then drop the individual columns.
    present_md = [c for c in MARKDOWN_COLS if c in df]
    df["TotalMarkDown"] = df[present_md].sum(axis=1) if present_md else 0.0
    df = df.drop(columns=present_md)

    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df


def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Drop rows whose Weekly_Sales is a z-score outlier within its
    (Store, Dept) group, mirroring the store/department-wise filtering."""
    def _mask(group: pd.Series) -> pd.Series:
        if len(group) < 3 or group.std(ddof=0) == 0:
            return pd.Series(True, index=group.index)
        return np.abs(stats.zscore(group)) < z_thresh

    keep = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .transform(lambda g: _mask(g))
        .astype(bool)
    )
    return df.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabular view (Random Forest / ANN)
# ---------------------------------------------------------------------------
@dataclass
class FeatureMatrix:
    X: pd.DataFrame
    y: pd.Series
    target_scaler: MinMaxScaler
    selected_features: list[str]


def build_feature_matrix(df: pd.DataFrame | None = None) -> FeatureMatrix:
    """One-hot encode, min-max scale the target and run RFE feature selection."""
    if df is None:
        df = remove_outliers(clean_and_merge(load_raw()))

    df = df.copy()

    # Scale the target to [0, 1] — metrics in the paper are on the scaled target.
    target_scaler = MinMaxScaler()
    y = pd.Series(
        target_scaler.fit_transform(df[["Weekly_Sales"]]).ravel(),
        index=df.index,
        name="Weekly_Sales",
    )

    # One-hot encode the categorical columns. dtype=int keeps the dummies as
    # numeric (the pandas default is bool, which select_dtypes(number) drops).
    cat_cols = [c for c in ("Store", "Dept", "Type") if c in df]
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    drop_cols = ["Weekly_Sales", "Date"]
    X = encoded.drop(columns=[c for c in drop_cols if c in encoded])
    X = X.select_dtypes(include=[np.number, bool]).astype(float).fillna(0)

    # Recursive Feature Elimination using a small forest as the estimator.
    # Feature *selection* doesn't need the full dataset or many trees, and
    # eliminating in chunks (step=0.2) keeps the number of refits small — so
    # RFE finishes in seconds instead of grinding through ~125 one-at-a-time
    # refits on 400k+ rows.
    n_features = min(config.RFE_N_FEATURES, X.shape[1])
    if len(X) > config.RFE_SAMPLE_ROWS:
        idx = X.sample(config.RFE_SAMPLE_ROWS, random_state=config.SEED).index
        X_sel, y_sel = X.loc[idx], y.loc[idx]
    else:
        X_sel, y_sel = X, y
    selector = RFE(
        RandomForestRegressor(n_estimators=30, random_state=config.SEED, n_jobs=-1),
        n_features_to_select=n_features,
        step=0.2,
    )
    selector.fit(X_sel, y_sel)
    selected = X.columns[selector.support_].tolist()

    return FeatureMatrix(X[selected], y, target_scaler, selected)


# ---------------------------------------------------------------------------
# Weekly series (time-series models + hybrids)
# ---------------------------------------------------------------------------
def build_weekly_series(df: pd.DataFrame | None = None) -> pd.Series:
    """Aggregate total weekly sales into a single scaled time series (``df_week``)."""
    if df is None:
        df = remove_outliers(clean_and_merge(load_raw()))

    weekly = (
        df.groupby("Date")["Weekly_Sales"].sum().sort_index().asfreq("W", method="ffill")
    )
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(weekly.values.reshape(-1, 1)).ravel()
    series = pd.Series(scaled, index=weekly.index, name="Weekly_Sales")
    series.attrs["scaler"] = scaler
    return series


def train_test_split_series(series: pd.Series, test_size: float = config.TEST_SIZE):
    """Chronological split — the last ``test_size`` fraction is the test horizon."""
    n_test = max(1, int(len(series) * test_size))
    return series.iloc[:-n_test], series.iloc[-n_test:]


def make_lag_features(series: pd.Series, n_lags: int = config.N_LAGS) -> pd.DataFrame:
    """Supervised view of the weekly series for the ML models (Random Forest /
    ANN), so every model is evaluated on the *same* forecasting task with a
    chronological split — no time leakage. Features use only past values.
    """
    feats = pd.DataFrame(index=series.index)
    feats["y"] = series.values
    for lag in range(1, n_lags + 1):
        feats[f"lag{lag}"] = series.shift(lag)
    feats["roll_mean"] = series.shift(1).rolling(n_lags).mean()
    feats["roll_std"] = series.shift(1).rolling(n_lags).std()
    feats["month"] = series.index.month
    feats["week"] = series.index.isocalendar().week.astype(int)
    return feats.fillna(0.0)
