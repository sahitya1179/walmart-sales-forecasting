"""ARIMA forecaster.

Order ``(p, d, q)`` is selected automatically with ``pmdarima.auto_arima``
(AIC + stationarity diagnostics), after a Dickey-Fuller stationarity check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .. import config


def is_stationary(series: pd.Series, alpha: float = 0.05) -> bool:
    """Augmented Dickey-Fuller test. True => series is stationary."""
    try:
        return adfuller(series.dropna())[1] < alpha
    except Exception:
        return False


def fit_auto_arima(train: pd.Series):
    """Fit a seasonal ARIMA whose order is chosen by ``auto_arima``."""
    import pmdarima as pm

    return pm.auto_arima(
        train,
        seasonal=True,
        m=config.SEASONAL_PERIOD,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        random_state=config.SEED,
    )


def forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    """Train on ``train`` and forecast ``len(test)`` steps ahead."""
    model = fit_auto_arima(train)
    return np.asarray(model.predict(n_periods=len(test)), dtype=float)
