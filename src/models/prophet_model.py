"""Facebook/Meta Prophet forecaster."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_prophet_frame(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"ds": series.index, "y": series.values})


def forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    """Fit Prophet on the training weeks and predict the test weeks."""
    from prophet import Prophet

    model = Prophet(weekly_seasonality=False, daily_seasonality=False,
                    yearly_seasonality=True)
    model.fit(_to_prophet_frame(train))

    future = pd.DataFrame({"ds": test.index})
    forecast_df = model.predict(future)
    return np.asarray(forecast_df["yhat"].values, dtype=float)
