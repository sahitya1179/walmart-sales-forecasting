"""Proposed ARIMA-RF hybrid model.

Two stages:

1. **ARIMA** models the linear trend/seasonality of the weekly series. From the
   fitted model we extract ARIMA-derived features via ``create_arima_features``:
   the ARIMA prediction, its first/second differences and the lagged residual.
2. **Random Forest** is trained on those ARIMA features (the original sales
   value at time *t* is excluded to avoid leakage) and produces the final
   forecast, learning the nonlinear residual structure ARIMA cannot capture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .. import config
from .arima_model import fit_auto_arima


def create_arima_features(series: pd.Series, model, train_len: int) -> pd.DataFrame:
    """Build the ARIMA-derived feature frame for the whole series.

    In-sample ARIMA predictions cover the training span; out-of-sample
    forecasts cover the test span. All features are functions of the ARIMA
    output, the calendar, or *lagged* actuals — never the current target.
    """
    n_test = len(series) - train_len
    in_sample = np.asarray(model.predict_in_sample())
    # Guard against length mismatches from differencing warm-up.
    if len(in_sample) != train_len:
        in_sample = np.resize(in_sample, train_len)
    out_sample = np.asarray(model.predict(n_periods=n_test))
    arima_pred = pd.Series(
        np.concatenate([in_sample, out_sample]), index=series.index, name="arima_pred"
    )

    feats = pd.DataFrame(index=series.index)
    feats["arima_pred"] = arima_pred
    feats["arima_diff1"] = arima_pred.diff()
    feats["arima_diff2"] = arima_pred.diff().diff()
    feats["resid_lag1"] = (series - arima_pred).shift(1)   # past residual only
    feats["month"] = series.index.month
    feats["week"] = series.index.isocalendar().week.astype(int)
    return feats.fillna(0.0)


def forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    """Fit ARIMA on ``train``, then a Random Forest on the ARIMA features."""
    full = pd.concat([train, test])
    train_len = len(train)

    model = fit_auto_arima(train)
    feats = create_arima_features(full, model, train_len)

    X_train, X_test = feats.iloc[:train_len], feats.iloc[train_len:]
    y_train = train.values

    rf = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS, random_state=config.SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return np.asarray(rf.predict(X_test), dtype=float)
