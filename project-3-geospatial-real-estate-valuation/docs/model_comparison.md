# Model Comparison: XGBoost Baseline vs Spatial Attention Model

All metrics computed on the same **20% holdout split** (random_state=42).

## Results

| Model | MAPE | RMSE |
|-------|------|------|
| XGBoost Baseline | 15.59% | 122,365 |
| Spatial Attention | 13.77% | 113,751 |

## Verdict

✅ **Spatial model outperforms XGBoost** — MAPE improved by **11.65%** (15.59% → 13.77%).

This confirms that incorporating neighborhood graph context via attention adds predictive value beyond standard tabular features.

## Regenerate

```powershell
python scripts/train_spatial_model.py
python scripts/compare_models.py
```