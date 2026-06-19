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


def prepare_holdout_frame(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
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

    return train_test_split(
        X,
        y,
        meta,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
    )


def compute_holdout_errors(features_df: pd.DataFrame) -> pd.DataFrame:
    """Train the baseline model and return holdout predictions with row-level errors."""
    import xgboost as xgb

    X_train, X_test, y_train, y_test, _meta_train, meta_test = prepare_holdout_frame(features_df)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        random_state=HOLDOUT_RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    errors = meta_test.copy()
    errors["actual_price"] = y_test.values
    errors["predicted_price"] = predictions
    errors["absolute_error"] = (errors["actual_price"] - errors["predicted_price"]).abs()
    errors["percentage_error"] = errors["absolute_error"] / errors["actual_price"].clip(lower=1)
    return errors.reset_index(drop=True)


def summarize_errors_by_zipcode(error_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate holdout errors at the ZIP-code level."""
    summary = (
        error_df.groupby("zipcode", as_index=False)
        .agg(
            listings=("actual_price", "count"),
            mean_actual_price=("actual_price", "mean"),
            mape=("percentage_error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            mean_neighbor_ratio=("sqft_living_vs_neighbor_ratio", "mean"),
            renovation_rate=("is_renovated", "mean"),
        )
        .sort_values("mape", ascending=False)
        .reset_index(drop=True)
    )
    summary["mape_pct"] = summary["mape"] * 100
    return summary


def identify_limitation_neighborhoods(
    zip_summary: pd.DataFrame,
    high_mape_quantile: float = 0.90,
    gentrification_mape_quantile: float = 0.75,
    min_listings: int = 20,
) -> dict[str, pd.DataFrame]:
    """Flag ZIP codes where tabular XGBoost struggles most.

    Gentrification proxy: elevated MAPE combined with higher-than-median
    renovation activity and living-area mismatch vs neighbors.
    """
    eligible = zip_summary[zip_summary["listings"] >= min_listings].copy()
    if eligible.empty:
        raise ValueError("No ZIP codes meet the minimum listing threshold for analysis.")

    mape_threshold = eligible["mape"].quantile(high_mape_quantile)
    gentrification_mape_threshold = eligible["mape"].quantile(gentrification_mape_quantile)
    renovation_threshold = eligible["renovation_rate"].median()
    neighbor_ratio_threshold = eligible["mean_neighbor_ratio"].median()

    high_error = eligible[eligible["mape"] >= mape_threshold].sort_values("mape", ascending=False)
    gentrifying = eligible[
        (eligible["mape"] >= gentrification_mape_threshold)
        & (eligible["renovation_rate"] >= renovation_threshold)
        & (eligible["mean_neighbor_ratio"] >= neighbor_ratio_threshold)
    ].sort_values("mape", ascending=False)
    luxury_stress = eligible[
        (eligible["mean_actual_price"] >= eligible["mean_actual_price"].quantile(0.90))
        & (eligible["mape"] >= eligible["mape"].median())
    ].sort_values("mean_actual_price", ascending=False)

    return {
        "high_error_zipcodes": high_error,
        "gentrification_proxy_zipcodes": gentrifying,
        "luxury_market_zipcodes": luxury_stress,
        "thresholds": pd.DataFrame(
            [
                {
                    "high_mape_threshold_pct": mape_threshold * 100,
                    "gentrification_mape_threshold_pct": gentrification_mape_threshold * 100,
                    "renovation_rate_threshold": renovation_threshold,
                    "neighbor_ratio_threshold": neighbor_ratio_threshold,
                    "min_listings": min_listings,
                }
            ]
        ),
    }
