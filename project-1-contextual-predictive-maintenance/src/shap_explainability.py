"""Utilities for generating SHAP-based feature importance explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_importance_table(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Summarize absolute SHAP values by feature and sort them descending."""
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(-1, 1)

    if shap_array.shape[1] != len(feature_names):
        raise ValueError("Number of feature names must match the SHAP value columns")

    mean_abs = np.mean(np.abs(shap_array), axis=0)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.reset_index(drop=True, inplace=True)
    return importance
