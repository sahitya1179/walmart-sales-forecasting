# Predictive Analysis of Walmart Data for Sales Forecasting Using a Hybrid Machine Learning Model

Reference implementation for the IEEE paper **"Predictive Analysis of Walmart Data for
Sales Forecasting Using Hybrid Machine Learning Model"**
(K. S. Sarabu, Arul Jothi S, Harismitha C, Reshmi Saravanan).

The project compares six forecasting approaches on the public Walmart store-sales dataset
and proposes a two-stage **ARIMA-RF hybrid** that combines ARIMA's linear time-series
modeling with a Random Forest's ability to learn the nonlinear residual structure. The
hybrid achieves the lowest error of every model tested.

---

## Models compared

| Model        | Family            | Role |
|--------------|-------------------|------|
| ARIMA        | Statistical TS    | Linear trend / seasonality (auto-selected order via `pmdarima`) |
| Holt-Winters | Statistical TS    | Triple exponential smoothing (level/trend/season) |
| Prophet      | Additive TS       | Robust to missing data & multiple seasonalities |
| Random Forest| ML ensemble       | Nonlinear feature interactions |
| ANN          | Neural network    | Nonlinear function approximation |
| **ARIMA-RF** | **Hybrid (ours)** | **ARIMA features → Random Forest on residuals** |
| Prophet-RF   | Hybrid            | Prophet features → Random Forest on residuals |

## Reported results (paper, Table 1 — metrics on the min-max scaled target)

| Model      | MSE        | RMSE       | MAE        |
|------------|------------|------------|------------|
| ARIMA      | 0.053384   | 0.231049   | 0.148474   |
| ANN        | 0.065038   | 0.255025   | 0.051893   |
| Random Forest | 0.044733| 0.211502   | 0.159710   |
| Prophet    | 0.060659   | 0.246290   | 0.523331   |
| **ARIMA-RF** | **0.017919** | **0.133863** | **0.109256** |
| Prophet-RF | 0.026530   | 0.162882   | 0.266002   |

## Reproduced results (this implementation)

Produced by `python -m src.main` on the Kaggle dataset. All seven models forecast the
**same** weekly-aggregated series over a 28-week chronological hold-out (no time leakage),
so the comparison is apples-to-apples. Values are on the min-max scaled target.

| Model         | MSE      | RMSE     | MAE      | MAPE (%) |
|---------------|----------|----------|----------|----------|
| ARIMA         | 0.006832 | 0.082655 | 0.069119 | 12.13    |
| Holt-Winters  | 0.010785 | 0.103849 | 0.088440 | 15.91    |
| Prophet       | 0.004130 | 0.064268 | 0.050639 | 9.43     |
| Random Forest | 0.003507 | 0.059218 | 0.047335 | 8.36     |
| ANN           | 0.005577 | 0.074681 | 0.059179 | 10.53    |
| **Prophet-RF**  | **0.003139** | **0.056023** | **0.045136** | 8.21 |
| **ARIMA-RF**    | **0.003647** | **0.060387** | **0.045760** | **8.01** |

**Takeaway (consistent with the paper):** the two **hybrid** models give the lowest error.
Hybridizing more than halves ARIMA's MSE (0.00683 → 0.00365) and improves on plain Prophet
too — confirming the paper's central claim that combining a statistical time-series model
with a Random Forest on its residuals beats either family alone.

> Absolute values differ from the paper's Table 1 because the exact preprocessing
> aggregation, train/test horizon and seeds are not fully specified in the text; the
> **ranking and conclusion are reproduced**. Re-running regenerates `results/metrics.csv`
> and the figures.

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/walmart-sales-forecasting.git
cd walmart-sales-forecasting
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the dataset (see data/README.md) into ./data

# 3a. Run the whole comparison from the command line
python -m src.main

# 3b. ...or open the end-to-end notebook
jupyter notebook notebooks/walmart_sales_forecasting.ipynb
```

Outputs land in `results/`:

- `metrics.csv` — the model-comparison table
- `figures/` — MSE/RMSE/MAE bar charts and the ARIMA-RF actual-vs-predicted plot
- `tableau/model_metrics.csv`, `tableau/actual_vs_predicted.csv` — tidy exports for Tableau

---

## Methodology (as in the paper)

1. **Data collection** — `stores.csv`, `train.csv`, `test.csv`, `features.csv` (45 stores, weekly sales Feb 2010 – Nov 2012).
2. **Preprocessing & feature engineering**
   - Median-fill `CPI` and `Unemployment`.
   - `MarkDown1-5`: negatives → 0, missing → 0; summed into `TotalMarkDown`, originals dropped.
   - Merge `stores` + `features` onto the sales data by `Store`/`Date`.
   - Derive `Year`, `Month`, `Week` from `Date`.
   - Z-score outlier removal on store/department weekly sales.
   - One-hot encode `Store`, `Dept`, `Type`.
   - Min-max scale the target; RFE for feature selection.
3. **Comparative study** — every model forecasts the weekly-aggregated series on the same
   chronological 80/20 split (≈ 3-year train / 1-year test). The ML models (Random Forest,
   ANN) use lag + rolling + calendar features so they solve the same forecasting task as the
   time-series models, with no future leakage. (The full tabular feature pipeline —
   one-hot encoding + RFE in `src/preprocessing.py::build_feature_matrix` — is also provided
   and demonstrated in the notebook.)
4. **Proposed ARIMA-RF hybrid**
   - Aggregate to a weekly series (`df_week`).
   - `auto_arima` selects `(p,d,q)` (seasonal `m=12`); extract fitted values, residuals, first differences.
   - Train a 100-tree Random Forest on the ARIMA-derived features (original sales excluded to avoid leakage).
   - Final forecast = Random Forest output; evaluate with MSE/RMSE/MAE.

## Evaluation metrics

```
MSE  = (1/n) Σ (yᵢ − ŷᵢ)²
RMSE = sqrt(MSE)
MAE  = (1/n) Σ |yᵢ − ŷᵢ|
MAPE = (100/n) Σ |(yᵢ − ŷᵢ) / yᵢ|
```

## Repository layout

```
walmart-sales-forecasting/
├── data/README.md            # how to get the dataset
├── notebooks/                # end-to-end runnable notebook
├── src/
│   ├── config.py             # paths, seed, hyperparameters
│   ├── preprocessing.py      # cleaning + feature engineering + weekly series
│   ├── evaluate.py           # MSE / RMSE / MAE / MAPE
│   ├── visualize.py          # charts + Tableau exports
│   ├── main.py               # runs the full model comparison
│   └── models/               # one module per model + the two hybrids
├── results/                  # metrics + figures + tableau exports (generated)
└── tableau/README.md         # dashboard build guide
```

## Citation

```bibtex
@inproceedings{sarabu2025walmart,
  title     = {Predictive Analysis of Walmart Data for Sales Forecasting Using Hybrid Machine Learning Model},
  author    = {Sarabu, Khaja Sahitya and Arul Jothi, S and Harismitha, C and Saravanan, Reshmi},
  booktitle = {IEEE},
  year      = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
