"""Prophet-RF hybrid model (baseline hybrid compared against ARIMA-RF).

Same two-stage idea as the ARIMA-RF hybrid, but Prophet supplies the linear /
seasonal base prediction that the Random Forest then refines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .. import config


def _prophet_predictions(train: pd.Series, full_index: pd.DatetimeIndex) -> pd.Series:
    from prophet import Prophet

    model = Prophet(weekly_seasonality=False, daily_seasonality=False,
                    yearly_seasonality=True)
    model.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    pred = model.predict(pd.DataFrame({"ds": full_index}))
    return pd.Series(pred["yhat"].values, index=full_index, name="prophet_pred")


def create_prophet_features(series: pd.Series, prophet_pred: pd.Series) -> pd.DataFrame:
    feats = pd.DataFrame(index=series.index)
    feats["prophet_pred"] = prophet_pred
    feats["prophet_diff1"] = prophet_pred.diff()
    feats["prophet_diff2"] = prophet_pred.diff().diff()
    feats["resid_lag1"] = (series - prophet_pred).shift(1)
    feats["month"] = series.index.month
    feats["week"] = series.index.isocalendar().week.astype(int)
    return feats.fillna(0.0)


def forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    full = pd.concat([train, test])
    train_len = len(train)

    prophet_pred = _prophet_predictions(train, full.index)
    feats = create_prophet_features(full, prophet_pred)

    X_train, X_test = feats.iloc[:train_len], feats.iloc[train_len:]
    rf = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS, random_state=config.SEED, n_jobs=-1
    )
    rf.fit(X_train, train.values)
    return np.asarray(rf.predict(X_test), dtype=float)
