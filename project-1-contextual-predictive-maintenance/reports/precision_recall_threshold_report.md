# Precision-Recall Threshold Tuning

This report captures the precision-recall behavior of the predictive maintenance classifier and documents the threshold chosen to balance false alarms with missed maintenance windows.

## Setup

- Dataset: data\processed\fused_dataset.csv
- Model: LightGBM classifier trained on a 75/25 train-test split.
- Metric: Average precision and threshold-based precision/recall tradeoff.

## Results

- Average precision: 0.9829
- Chosen threshold: 0.1500
- Precision at chosen threshold: 1.0000
- Recall at chosen threshold: 0.9765
- F1 at chosen threshold: 0.9881
- False positive rate at chosen threshold: 0.0000
- False negative rate at chosen threshold: 0.0235

## Saved Artifacts

- PR curve data: reports\precision_recall_curve.csv
- PR curve plot: reports\precision_recall_curve.png

The chosen threshold is the point that preserves strong recall while minimizing the weighted false-alarm versus missed-maintenance cost.