"""Central configuration: paths, random seed and shared hyperparameters.

Keeping these in one place makes every run reproducible and lets the notebook
and the CLI (`python -m src.main`) share identical settings.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we touch so results are deterministic across runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # TensorFlow is optional at import time
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # pragma: no cover - tf may be absent
        pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

STORES_CSV = DATA_DIR / "stores.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
FEATURES_CSV = DATA_DIR / "features.csv"

for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Modeling hyperparameters (as described in the paper)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20          # 80/20 train/test split (~3yr train, ~1yr test)
SEASONAL_PERIOD = 12      # weekly series seasonal period used by the hybrid
RF_N_ESTIMATORS = 100     # Random Forest trees
RFE_N_FEATURES = 15       # features kept by Recursive Feature Elimination
RFE_SAMPLE_ROWS = 40000   # row subsample used only for feature selection (speed)
N_LAGS = 4                # lag features for the ML models on the weekly series

# Order in which models appear in tables / charts.
MODEL_ORDER = [
    "ARIMA",
    "Holt-Winters",
    "Prophet",
    "Random Forest",
    "ANN",
    "Prophet-RF",
    "ARIMA-RF",
]
