import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
import xgboost as xgb


def train_valuation_model(features_df: pd.DataFrame, model_save_path: str):
    """Train an XGBoost regressor on tabular features.

    Parameters
    ----------
    features_df: pd.DataFrame
        DataFrame containing engineered features and the target column ``price_normalized``.
    model_save_path: str
        File path where the trained model will be persisted (e.g. ``models/xgboost_regressor.pkl``).

    Returns
    -------
    dict
        Dictionary with ``mape`` and ``rmse`` values on the hold‑out set.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Drop non-numeric / geometry columns and other price columns to prevent leakage
    leakage_cols = ["geometry", "lat", "long", "zipcode", "id", "date", "price", "price_upper_cap", "is_price_outlier"]
    drop_cols = [col for col in leakage_cols if col in features_df.columns]
    df = features_df.drop(columns=drop_cols)

    # Target column must exist
    if "price_normalized" not in df.columns:
        raise KeyError("'price_normalized' column not found in features DataFrame")

    X = df.drop(columns=["price_normalized"])
    y = df["price_normalized"]

    # Train‑test split (80 % train / 20 % hold‑out) with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialise XGBoost regressor with default baseline hyper‑parameters
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train)

    # Predictions on hold‑out set
    preds = model.predict(X_test)

    # Evaluation metrics
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"Baseline XGBoost – MAPE: {mape:.4f}, RMSE: {rmse:.4f}")

    # Persist the model
    joblib.dump(model, model_save_path)

    return {"mape": mape, "rmse": rmse}

