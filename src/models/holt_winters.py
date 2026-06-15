"""Holt-Winters (triple exponential smoothing) forecaster."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .. import config


def forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    """Additive trend + additive seasonality exponential smoothing."""
    seasonal_periods = config.SEASONAL_PERIOD
    # Need at least two full seasons to estimate a seasonal component.
    use_seasonal = len(train) >= 2 * seasonal_periods
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add" if use_seasonal else None,
        seasonal_periods=seasonal_periods if use_seasonal else None,
        initialization_method="estimated",
    ).fit()
    return np.asarray(model.forecast(len(test)), dtype=float)
