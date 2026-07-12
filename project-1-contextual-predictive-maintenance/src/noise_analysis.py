"""Utilities for evaluating model robustness to synthetic sensor noise."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def inject_sensor_noise(
    df: pd.DataFrame,
    noise_level: float,
    sensor_columns: list[str],
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add Gaussian noise to selected sensor columns in a copy of the dataframe."""
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    noisy_df = df.copy()
    if noise_level == 0:
        return noisy_df

    rng = np.random.default_rng(random_state)
    for column in sensor_columns:
        if column not in noisy_df.columns:
            raise KeyError(f"Column {column} not found in dataframe")
        scale = noise_level * noisy_df[column].std(ddof=0)
        if scale == 0:
            scale = noise_level
        noisy_df[column] = noisy_df[column] + rng.normal(0, scale, size=len(noisy_df))
    return noisy_df


def evaluate_noise_sensitivity(
    df: pd.DataFrame,
    target_column: str,
    sensor_columns: list[str],
    noise_levels: list[float],
    random_state: int | None = None,
) -> pd.DataFrame:
    """Train a simple classifier on noisy variants of the data and report Macro F1."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    results = []
    for noise_level in noise_levels:
        x_test_noisy = inject_sensor_noise(
            X_test.copy(),
            noise_level=noise_level,
            sensor_columns=sensor_columns,
            random_state=random_state,
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        model.fit(X_train, y_train)
        predictions = model.predict(x_test_noisy)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        results.append({"noise_level": noise_level, "macro_f1": macro_f1})

    return pd.DataFrame(results)
