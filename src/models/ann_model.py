"""Artificial Neural Network (ANN) forecaster on the weekly series.

Uses a small Keras MLP when TensorFlow is available, and falls back to
scikit-learn's ``MLPRegressor`` otherwise so the pipeline always runs. Features
are standardized (neural nets are scale-sensitive). Evaluated on the same
weekly series and chronological split as every other model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .. import config
from ..preprocessing import make_lag_features


def _keras_mlp(input_dim: int):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    full = pd.concat([train, test])
    train_len = len(train)
    feats = make_lag_features(full)

    X = feats.drop(columns="y").values
    y = feats["y"].values
    X_train, X_test = X[:train_len], X[train_len:]
    y_train = y[:train_len]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    try:  # Preferred: a real (Keras) neural network.
        config.set_seed()
        model = _keras_mlp(X_train.shape[1])
        model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)
        return model.predict(X_test, verbose=0).ravel().astype(float)
    except Exception:  # TensorFlow missing/unavailable -> sklearn fallback.
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=config.SEED
        )
        model.fit(X_train, y_train)
        return np.asarray(model.predict(X_test), dtype=float)
