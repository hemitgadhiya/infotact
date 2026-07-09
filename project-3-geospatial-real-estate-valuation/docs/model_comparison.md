# Model Comparison: XGBoost vs Spatial Attention vs GNN

All metrics computed on the same **20% holdout split** (random_state=42).

## Results

| Model | MAPE | RMSE | Status |
|-------|------|------|--------|
| XGBoost Baseline | 15.59% | 122,365 | Trained & Evaluated |
| Spatial Attention (Manual) | 13.77% | 113,751 | Trained & Evaluated |
| GNN (Graph Attention Network) | 15.29% | 117,625 | Trained & Evaluated |

## Verdict

✅ **Spatial Attention (Manual)** is the top-performing model, achieving **13.77% MAPE**.
This represents a **11.65% relative improvement** over the tabular XGBoost baseline.

## Regenerate

```powershell
python scripts/train_spatial_model.py
python scripts/train_gnn_model.py
python scripts/compare_models.py
```