"""Plots for the model comparison."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd

from . import config


def _bar(metrics: pd.DataFrame, column: str, fname: str) -> None:
    ordered = metrics.sort_values(column)
    colors = ["#E08A12" if m == "ARIMA-RF" else "#9AA0A8" for m in ordered["Model"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ordered["Model"], ordered[column], color=colors)
    ax.set_title(f"{column} by Model (lower is better)")
    ax.set_ylabel(column)
    ax.set_xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame) -> None:
    """Fig. 6/7/8 — MSE, RMSE and MAE bar charts across models."""
    _bar(metrics, "MSE", "mse_vs_model.png")
    _bar(metrics, "RMSE", "rmse_vs_model.png")
    _bar(metrics, "MAE", "mae_vs_model.png")


def plot_actual_vs_predicted(index, y_true, y_pred,
                             title: str = "ARIMA-RF: Actual vs Predicted",
                             fname: str = "arima_rf_actual_vs_predicted.png") -> None:
    """Fig. 5 — actual vs predicted weekly sales for the hybrid."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(index, y_true, label="Actual", color="#23262C", linewidth=2)
    ax.plot(index, y_pred, label="Predicted", color="#E08A12", linewidth=2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly sales (scaled)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / fname, dpi=150)
    plt.close(fig)
