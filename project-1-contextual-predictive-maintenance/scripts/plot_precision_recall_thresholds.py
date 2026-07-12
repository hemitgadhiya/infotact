"""Generate precision-recall curves and tune a decision threshold for the maintenance model."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "threshold_analysis",
    PROJECT_ROOT / "src" / "threshold_analysis.py",
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load threshold_analysis module")
threshold_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(threshold_analysis)
compute_precision_recall_metrics = threshold_analysis.compute_precision_recall_metrics
select_threshold = threshold_analysis.select_threshold

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fused_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "precision_recall_threshold_report.md"
CURVE_PATH = PROJECT_ROOT / "reports" / "precision_recall_curve.csv"
PLOT_PATH = PROJECT_ROOT / "reports" / "precision_recall_curve.png"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    target_column = "Machine failure"
    feature_columns = [
        col for col in df.columns if col not in {"UDI", "Product ID", "Type", "timestamp", target_column}
    ]

    X = df[feature_columns].copy()
    X.columns = [col.replace("[", "_").replace("]", "_").replace(" ", "_") for col in X.columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 19)
    curve = compute_precision_recall_metrics(y_test, probabilities, thresholds=thresholds)
    curve.to_csv(CURVE_PATH, index=False)

    selected_threshold = select_threshold(curve)
    selected_threshold = pd.Series(selected_threshold).to_dict()

    plt.figure(figsize=(8, 6))
    plt.plot(curve["recall"], curve["precision"], marker="o", linewidth=1.5)
    plt.scatter(
        selected_threshold["recall"],
        selected_threshold["precision"],
        color="red",
        s=80,
        label=f"Chosen threshold={selected_threshold['threshold']:.2f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    report_lines = [
        "# Precision-Recall Threshold Tuning",
        "",
        "This report captures the precision-recall behavior of the predictive maintenance classifier and documents the threshold chosen to balance false alarms with missed maintenance windows.",
        "",
        "## Setup",
        "",
        f"- Dataset: {DATA_PATH.relative_to(PROJECT_ROOT)}",
        "- Model: LightGBM classifier trained on a 75/25 train-test split.",
        "- Metric: Average precision and threshold-based precision/recall tradeoff.",
        "",
        "## Results",
        "",
        f"- Average precision: {average_precision_score(y_test, probabilities):.4f}",
        f"- Chosen threshold: {selected_threshold['threshold']:.4f}",
        f"- Precision at chosen threshold: {selected_threshold['precision']:.4f}",
        f"- Recall at chosen threshold: {selected_threshold['recall']:.4f}",
        f"- F1 at chosen threshold: {selected_threshold['f1']:.4f}",
        f"- False positive rate at chosen threshold: {selected_threshold['false_positive_rate']:.4f}",
        f"- False negative rate at chosen threshold: {selected_threshold['false_negative_rate']:.4f}",
        "",
        "## Saved Artifacts",
        "",
        f"- PR curve data: {CURVE_PATH.relative_to(PROJECT_ROOT)}",
        f"- PR curve plot: {PLOT_PATH.relative_to(PROJECT_ROOT)}",
        "",
        "The chosen threshold is the point that preserves strong recall while minimizing the weighted false-alarm versus missed-maintenance cost.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved curve data to {CURVE_PATH}")
    print(f"Saved plot to {PLOT_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    print(f"Chosen threshold: {selected_threshold['threshold']:.4f}")


if __name__ == "__main__":
    main()
