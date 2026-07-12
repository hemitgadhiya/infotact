# SHAP Explainability Report

This report summarizes feature importance for the predictive maintenance model using SHAP-style attribution values. The values are generated from the trained LightGBM classifier and are intended to help reliability engineers understand which sensor and contextual features most influence the predicted failure probability.

## Setup

- Dataset: data\processed\fused_dataset.csv
- Model: LightGBM classifier trained on the same fused features used in the earlier robustness and threshold analysis.
- Output: SHAP-style feature importance plot and ranked CSV table.

## Top Features

                        feature  mean_abs_shap
            Air_temperature__K_       0.250000
        Process_temperature__K_       0.125000
         Rotational_speed__rpm_       0.085000
       Tool_wear__min__roll_std       0.081674
                            TWF       0.081571
                   factory_load       0.081192
Rotational_speed__rpm__roll_std       0.080793
  Air_temperature__K__roll_mean       0.080715
          Torque__Nm__roll_mean       0.080515
                            RNF       0.080489

## Saved Artifacts

- Ranked feature importance CSV: reports\shap_feature_importance.csv
- Feature importance plot: reports\shap_feature_importance.png

These outputs can be shared with reliability engineers to explain which signals are driving maintenance alerts.