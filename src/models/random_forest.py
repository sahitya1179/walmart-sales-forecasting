"""Random Forest forecaster on the weekly series.

To keep the comparison fair, the Random Forest forecasts the *same* weekly
series as the time-series models, using lag + calendar features and a
chronological split (no future leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .. import config
from ..preprocessing import make_lag_features


def forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    full = pd.concat([train, test])
    train_len = len(train)
    feats = make_lag_features(full)

    X = feats.drop(columns="y")
    y = feats["y"]
    X_train, X_test = X.iloc[:train_len], X.iloc[train_len:]

    model = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS, random_state=config.SEED, n_jobs=-1
    )
    model.fit(X_train, y.iloc[:train_len])
    return np.asarray(model.predict(X_test), dtype=float)
