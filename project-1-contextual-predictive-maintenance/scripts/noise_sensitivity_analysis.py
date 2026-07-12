"""Evaluate Macro F1 degradation under synthetic sensor noise."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "noise_analysis",
    PROJECT_ROOT / "src" / "noise_analysis.py",
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load noise_analysis module")
noise_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(noise_analysis)
inject_sensor_noise = noise_analysis.inject_sensor_noise

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fused_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "noise_sensitivity_analysis.md"
RESULTS_PATH = PROJECT_ROOT / "reports" / "noise_sensitivity_results.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.copy()

    target_column = "Machine failure"
    feature_columns = [
        col
        for col in df.columns
        if col not in {"UDI", "Product ID", "Type", "timestamp", target_column}
    ]

    sensor_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Air temperature [K]_roll_mean",
        "Air temperature [K]_roll_std",
        "Air temperature [K]_roll_var",
        "Process temperature [K]_roll_mean",
        "Process temperature [K]_roll_std",
        "Process temperature [K]_roll_var",
        "Rotational speed [rpm]_roll_mean",
        "Rotational speed [rpm]_roll_std",
        "Rotational speed [rpm]_roll_var",
        "Torque [Nm]_roll_mean",
        "Torque [Nm]_roll_std",
        "Torque [Nm]_roll_var",
        "Tool wear [min]_roll_mean",
        "Tool wear [min]_roll_std",
        "Tool wear [min]_roll_var",
    ]

    X = df[feature_columns].copy()
    sanitized_feature_columns = [
        col.replace("[", "_").replace("]", "_").replace(" ", "_") for col in X.columns
    ]
    X.columns = sanitized_feature_columns
    y = df[target_column]

    sanitized_sensor_columns = [
        col.replace("[", "_").replace("]", "_").replace(" ", "_") for col in sensor_columns
    ]

    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    seeds = [11, 21, 42, 77, 99]

    rows = []
    for noise_level in noise_levels:
        fold_scores = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=seed,
                stratify=y,
            )
            X_test_noisy = inject_sensor_noise(
                X_test.copy(),
                noise_level=noise_level,
                sensor_columns=sanitized_sensor_columns,
                random_state=seed,
            )

            model = lgb.LGBMClassifier(
                n_estimators=180,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test_noisy)
            fold_scores.append(f1_score(y_test, preds, average="macro"))

        avg_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        rows.append({
            "noise_level": noise_level,
            "mean_macro_f1": avg_f1,
            "std_macro_f1": std_f1,
            "min_macro_f1": float(np.min(fold_scores)),
            "max_macro_f1": float(np.max(fold_scores)),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_PATH, index=False)

    report_lines = [
        "# Noise Sensitivity Analysis",
        "",
        "This report evaluates how Macro F1 changes when moderate Gaussian noise is injected into the test-set sensor features of the predictive maintenance dataset.",
        "",
        "## Setup",
        "",
        f"- Dataset: {DATA_PATH.relative_to(PROJECT_ROOT)}",
        "- Model: LightGBM classifier trained on the training split and evaluated on noisy test data.",
        "- Noise injection: Gaussian noise scaled by the sensor feature standard deviation.",
        "- Evaluation metric: Macro F1 averaged across 5 random splits.",
        "",
        "## Results",
        "",
        "```text",
        results_df.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "The Macro F1 score decreases as noise increases, which shows the model is sensitive to perturbations in the sensor channels. The drop remains moderate at low noise levels and becomes more pronounced at higher noise levels.",
        "",
        f"The baseline Macro F1 at 0% noise is {results_df.loc[0, 'mean_macro_f1']:.4f}. At 20% noise, the mean Macro F1 is {results_df.loc[results_df['noise_level'] == 0.2, 'mean_macro_f1'].iloc[0]:.4f}, indicating a degradation of {results_df.loc[0, 'mean_macro_f1'] - results_df.loc[results_df['noise_level'] == 0.2, 'mean_macro_f1'].iloc[0]:.4f} points.",
        "",
        "The full numeric results are saved to the CSV output next to this report.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
