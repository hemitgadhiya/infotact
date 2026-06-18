# XGBoost Baseline Limitations by Neighborhood

This report summarizes where the tabular XGBoost baseline underperforms
before spatial graph modeling is introduced in Week 3.

## Overall holdout performance

- Overall holdout MAPE: **12.71%**
- ZIP codes analyzed: **70**

## Why the baseline fails in some areas

1. **Gentrification proxy areas** mix renovated homes with older neighbor comps.
2. **Luxury ZIP codes** have heavy price tails that tabular features underfit.
3. **Spatial spillovers** are missing because lat/long are dropped at training time.

## Detection thresholds

- High-error MAPE cutoff: 18.59%
- Gentrification MAPE cutoff: 14.76%
- Minimum listings per ZIP: 20

## Highest-error ZIP codes

| ZIP | Listings | MAPE % | Mean Price | Renovation Rate |
|-----|----------|--------|------------|-----------------|
| 98166 | 44 | 28.26 | $384,031 | 13.6% |
| 98146 | 49 | 23.61 | $418,440 | 8.2% |
| 98014 | 23 | 22.80 | $414,063 | 4.3% |
| 98168 | 61 | 21.31 | $240,239 | 4.9% |
| 98118 | 100 | 21.04 | $440,566 | 5.0% |

## Gentrification-proxy ZIP codes

| ZIP | Listings | MAPE % | Neighbor Ratio | Renovation Rate |
|-----|----------|--------|----------------|-----------------|
| 98014 | 23 | 22.80 | 1.11 | 4.3% |
| 98118 | 100 | 21.04 | 1.12 | 5.0% |
| 98112 | 56 | 19.07 | 1.10 | 14.3% |
| 98119 | 42 | 18.01 | 1.10 | 19.0% |
| 98106 | 70 | 17.44 | 1.09 | 4.3% |

## Luxury-market stress ZIP codes

| ZIP | Listings | MAPE % | Mean Price |
|-----|----------|--------|------------|
| 98004 | 57 | 13.94 | $1,346,781 |
| 98112 | 56 | 19.07 | $1,205,805 |
| 98040 | 70 | 14.92 | $1,179,274 |
| 98006 | 98 | 13.33 | $860,341 |
| 98033 | 85 | 15.98 | $832,746 |

## Implication for Week 3

These neighborhoods justify graph/spatial embeddings because local neighbor
context should explain price gaps that tabular features cannot capture alone.

Regenerate this report with:

```powershell
python scripts/analyze_baseline_errors.py
```
