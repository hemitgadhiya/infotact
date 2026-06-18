"""Analyze XGBoost baseline prediction errors by neighborhood."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

BASELINE_DROP_COLS = ["geometry", "lat", "long", "zipcode", "id", "date"]
HOLDOUT_META_COLS = [
    "zipcode",
    "lat",
    "long",
    "price",
    "price_normalized",
    "house_age",
    "is_renovated",
    "sqft_living_vs_neighbor_ratio",
    "dist_to_seattle_center_km",
]
HOLDOUT_RANDOM_STATE = 42
HOLDOUT_TEST_SIZE = 0.2


def prepare_holdout_frame(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split engineered features into train/test sets with holdout metadata."""
    if "price_normalized" not in features_df.columns:
        raise KeyError("'price_normalized' column not found in features DataFrame")

    drop_cols = [col for col in BASELINE_DROP_COLS if col in features_df.columns]
    meta_cols = [col for col in HOLDOUT_META_COLS if col in features_df.columns]

    feature_cols = [
        col
        for col in features_df.columns
        if col not in drop_cols and col not in meta_cols and col != "price_normalized"
    ]

    X = features_df[feature_cols]
    y = features_df["price_normalized"]
    meta = features_df[meta_cols]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        meta,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
    )

    return (
        pd.concat([X_train, y_train], axis=1),
        y_test,
        meta_test.reset_index(drop=True),
    )
