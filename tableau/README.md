# Tableau Dashboards

The pipeline exports two tidy CSVs to `results/tableau/` that feed three
dashboards. If you have the original `.twbx`, drop it in this folder and point
its data sources at those CSVs.

## Data sources

| File | Grain | Fields |
|------|-------|--------|
| `results/tableau/model_metrics.csv` | one row per model | `Model`, `MSE`, `RMSE`, `MAE`, `MAPE` |
| `results/tableau/actual_vs_predicted.csv` | one row per test week | `Week`, `Actual`, `Predicted` |

## Build steps

1. **Connect**: *Data ▸ New Data Source ▸ Text file* → select the CSV(s) above.
2. Create the worksheets below, then combine them on one dashboard.

### 1. Model comparison (MSE / RMSE / MAE)
- Source: `model_metrics.csv`
- Columns: `Model`; Rows: `MSE` (then duplicate sheets for `RMSE`, `MAE`).
- Sort ascending; color the `ARIMA-RF` bar with the accent (`#E08A12`) to show it wins.
- This reproduces **Fig. 6 / 7 / 8** from the paper.

### 2. Actual vs Predicted (ARIMA-RF)
- Source: `actual_vs_predicted.csv`
- Columns: `Week` (continuous); Rows: `Actual` and `Predicted` (dual / overlaid lines).
- Reproduces **Fig. 5**.

### 3. Forecast error KPI
- Source: `model_metrics.csv`
- Big number tiles for the `ARIMA-RF` MSE (0.0179), RMSE and MAE for an at-a-glance summary.

## Refreshing
Re-run `python -m src.main`, then in Tableau *Data ▸ Refresh* to pull the
regenerated CSVs.
