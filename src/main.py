"""End-to-end model comparison.

Run with::

    python -m src.main

Loads + preprocesses the Walmart data, trains every model, evaluates them on a
common test horizon, writes ``results/metrics.csv`` and the comparison figures.
"""
from __future__ import annotations

import sys
import traceback

import pandas as pd

from . import config
from .evaluate import evaluate
from .models import (
    ann_model,
    arima_model,
    arima_rf_hybrid,
    holt_winters,
    prophet_model,
    prophet_rf_hybrid,
    random_forest,
)
from .preprocessing import (
    build_weekly_series,
    clean_and_merge,
    load_raw,
    remove_outliers,
    train_test_split_series,
)
from .visualize import plot_actual_vs_predicted, plot_metric_bars


def run() -> pd.DataFrame:
    config.set_seed()

    print("Loading and preprocessing data ...")
    df = remove_outliers(clean_and_merge(load_raw()))

    # Weekly aggregated series — every model is evaluated on this same
    # forecasting task with one chronological train/test split, so the
    # comparison is fair (no model gets an easier problem or leaks the future).
    series = build_weekly_series(df)
    train_s, test_s = train_test_split_series(series)
    print(f"  weekly series: {len(series)} weeks "
          f"({len(train_s)} train / {len(test_s)} test)")

    results: dict[str, dict] = {}
    hybrid_pred = None  # kept for the actual-vs-predicted plot

    models = {
        "ARIMA": arima_model.forecast,
        "Holt-Winters": holt_winters.forecast,
        "Prophet": prophet_model.forecast,
        "Random Forest": random_forest.forecast,
        "ANN": ann_model.forecast,
        "Prophet-RF": prophet_rf_hybrid.forecast,
        "ARIMA-RF": arima_rf_hybrid.forecast,
    }
    for name, fn in models.items():
        try:
            print(f"Training {name} ...")
            y_pred = fn(train_s, test_s)
            results[name] = evaluate(test_s.values, y_pred)
            if name == "ARIMA-RF":
                hybrid_pred = y_pred
        except Exception as exc:  # one model failing must not kill the run
            print(f"  [skipped] {name}: {exc}")
            traceback.print_exc()

    # --- assemble + persist -------------------------------------------------
    metrics = (
        pd.DataFrame(results)
        .T.reset_index()
        .rename(columns={"index": "Model"})
    )
    order = [m for m in config.MODEL_ORDER if m in metrics["Model"].values]
    metrics["Model"] = pd.Categorical(metrics["Model"], categories=order, ordered=True)
    metrics = metrics.sort_values("Model").reset_index(drop=True)

    metrics_path = config.RESULTS_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print("\n=== Model comparison ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved metrics -> {metrics_path}")

    # --- figures ------------------------------------------------------------
    plot_metric_bars(metrics)
    if hybrid_pred is not None:
        plot_actual_vs_predicted(test_s.index, test_s.values, hybrid_pred)
    print(f"Saved figures -> {config.FIGURES_DIR}")

    return metrics


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
