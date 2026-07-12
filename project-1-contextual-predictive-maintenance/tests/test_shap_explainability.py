import numpy as np
import pandas as pd

from src.shap_explainability import build_feature_importance_table


def test_build_feature_importance_table_ranks_features():
    shap_values = np.array([[0.3, 0.1], [0.2, 0.8]])
    feature_names = ["sensor_a", "sensor_b"]

    importance = build_feature_importance_table(shap_values, feature_names)

    assert list(importance["feature"]) == ["sensor_b", "sensor_a"]
    assert importance.iloc[0]["mean_abs_shap"] >= importance.iloc[1]["mean_abs_shap"]
