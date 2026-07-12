# Noise Sensitivity Analysis

This report evaluates how Macro F1 changes when moderate Gaussian noise is injected into the test-set sensor features of the predictive maintenance dataset.

## Setup

- Dataset: data\processed\fused_dataset.csv
- Model: LightGBM classifier trained on the training split and evaluated on noisy test data.
- Noise injection: Gaussian noise scaled by the sensor feature standard deviation.
- Evaluation metric: Macro F1 averaged across 5 random splits.

## Results

```text
 noise_level  mean_macro_f1  std_macro_f1  min_macro_f1  max_macro_f1
        0.00       0.993164      0.005394      0.984331      1.000000
        0.05       0.990635      0.006305      0.981087      1.000000
        0.10       0.987508      0.004050      0.981087      0.990708
        0.15       0.983644      0.004800      0.974482      0.987538
        0.20       0.977060      0.006837      0.964273      0.984331
```

## Interpretation

The Macro F1 score decreases as noise increases, which shows the model is sensitive to perturbations in the sensor channels. The drop remains moderate at low noise levels and becomes more pronounced at higher noise levels.

The baseline Macro F1 at 0% noise is 0.9932. At 20% noise, the mean Macro F1 is 0.9771, indicating a degradation of 0.0161 points.

The full numeric results are saved to the CSV output next to this report.