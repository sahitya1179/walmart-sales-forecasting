# Dataset

This project uses the public **Walmart Recruiting – Store Sales Forecasting** dataset.

## Download

1. Get the data from Kaggle:
   <https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data>

   Or via the Kaggle CLI:
   ```bash
   pip install kaggle
   kaggle competitions download -c walmart-recruiting-store-sales-forecasting
   unzip walmart-recruiting-store-sales-forecasting.zip -d data/
   ```

2. Place these four files directly in this `data/` folder:

   ```
   data/
   ├── stores.csv
   ├── train.csv
   ├── test.csv
   └── features.csv
   ```

The CSVs are git-ignored, so they will not be committed to the repo.

## Schema

**stores.csv** — anonymized store metadata
| Column | Description |
|--------|-------------|
| `Store` | Store ID (1–45) |
| `Type`  | Store type (A/B/C) |
| `Size`  | Store size |

**train.csv** — historical weekly sales (5 Feb 2010 – 1 Nov 2012)
| Column | Description |
|--------|-------------|
| `Store` | Store ID |
| `Dept`  | Department ID |
| `Date`  | Week (date) |
| `Weekly_Sales` | Sales for that dept/store/week (target) |
| `IsHoliday` | Whether the week contains a holiday |

**test.csv** — same as `train.csv` but without `Weekly_Sales`.

**features.csv** — regional / promotional context per store & week
| Column | Description |
|--------|-------------|
| `Store` | Store ID |
| `Date`  | Week |
| `Temperature` | Avg regional temperature |
| `Fuel_Price`  | Regional fuel cost |
| `MarkDown1`–`MarkDown5` | Promotional markdowns (available after Nov 2011; `NA` otherwise) |
| `CPI` | Consumer price index |
| `Unemployment` | Regional unemployment rate |
| `IsHoliday` | Whether the week contains a holiday |
