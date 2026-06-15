"""Forecast-accuracy metrics used throughout the paper."""
from __future__ import annotations

import numpy as np


def mse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (%). Guards against divide-by-zero."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nonzero = y_true != 0
    if not nonzero.any():
        return float("nan")
    return float(
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    )


def evaluate(y_true, y_pred) -> dict[str, float]:
    """Return the full metric set for one model."""
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
