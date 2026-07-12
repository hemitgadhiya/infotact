"""Utilities for generating precision-recall curves and tuning decision thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score


def compute_precision_recall_metrics(
    y_true: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute precision, recall, F1, and cost metrics across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)
    if len(pr_thresholds) == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "cost_score"])

    rows = []
    for threshold in thresholds:
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (np.asarray(y_true) == 1))
        fp = np.sum((predictions == 1) & (np.asarray(y_true) == 0))
        fn = np.sum((predictions == 0) & (np.asarray(y_true) == 1))
        tn = np.sum((predictions == 0) & (np.asarray(y_true) == 0))

        precision_value = tp / (tp + fp) if (tp + fp) else 0.0
        recall_value = tp / (tp + fn) if (tp + fn) else 0.0
        f1_value = f1_score(y_true, predictions, average="binary", zero_division=0)
        false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
        cost_score = 0.7 * false_positive_rate + 0.3 * false_negative_rate

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_value),
                "recall": float(recall_value),
                "f1": float(f1_value),
                "false_positive_rate": float(false_positive_rate),
                "false_negative_rate": float(false_negative_rate),
                "cost_score": float(cost_score),
            }
        )

    return pd.DataFrame(rows)


def select_threshold(curve: pd.DataFrame) -> pd.Series:
    """Choose the threshold that minimizes a weighted cost while preserving strong recall."""
    if curve.empty:
        raise ValueError("curve must not be empty")

    best_row = curve.loc[curve["f1"].idxmax()]
    candidate_rows = curve[curve["recall"] >= 0.8]
    if not candidate_rows.empty:
        best_candidate = candidate_rows.loc[candidate_rows["cost_score"].idxmin()]
        if best_candidate["f1"] >= best_row["f1"] - 0.02:
            best_row = best_candidate

    return best_row
