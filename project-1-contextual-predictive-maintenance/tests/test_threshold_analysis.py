import pandas as pd

from src.threshold_analysis import compute_precision_recall_metrics, select_threshold


def test_compute_precision_recall_metrics_returns_threshold_rows():
    y_true = [0, 0, 1, 1]
    probabilities = [0.2, 0.4, 0.6, 0.8]

    curve = compute_precision_recall_metrics(y_true, probabilities, thresholds=[0.25, 0.5, 0.75])

    assert not curve.empty
    assert {"threshold", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "cost_score"}.issubset(curve.columns)
    assert list(curve["threshold"]) == [0.25, 0.5, 0.75]


def test_select_threshold_returns_one_of_the_thresholds():
    curve = pd.DataFrame(
        {
            "threshold": [0.3, 0.5, 0.7],
            "precision": [0.5, 1.0, 0.5],
            "recall": [1.0, 0.5, 0.5],
            "f1": [0.6667, 1.0, 0.5],
            "false_positive_rate": [0.5, 0.0, 0.5],
            "false_negative_rate": [0.0, 0.5, 0.5],
            "cost_score": [0.3, 0.25, 0.5],
        }
    )

    selected = select_threshold(curve)

    assert selected["threshold"] == 0.5
