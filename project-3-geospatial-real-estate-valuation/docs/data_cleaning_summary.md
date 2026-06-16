# King County Data Cleaning Summary

Pipeline: `run_pipeline.py`  
Source: [King County house sales dataset](https://raw.githubusercontent.com/jmatth11/King-County-House-Data-Set/master/kc_house_data.csv)

## Cleaning steps

1. Parse sale dates and normalize ID/ZIP types
2. Correct known 33-bedroom data entry typo to 3 bedrooms
3. Remove invalid rows (price <= $50k, zero bedrooms/bathrooms, out-of-bounds coordinates)
4. Deduplicate by property ID, keeping the latest sale
5. Build GeoDataFrame (EPSG:4326)
6. Winsorize extreme prices per ZIP code (Q3 + 3.0 * IQR cap)

## Latest run statistics

| Metric | Value |
|--------|-------|
| Raw rows | 21,436 |
| Clean rows | 21,420 |
| Rows removed in basic cleaning | 16 |
| Extreme price outliers capped | 352 (1.64%) |
| Mean price (original) | $541,766.86 |
| Mean price (normalized) | $534,421.13 |
| Max price (original) | $7,700,000.00 |
| Max price (normalized) | $6,070,000.00 |

## Outputs (local only, not committed)

- `data/raw/kc_house_data.csv`
- `data/processed/kc_house_data_cleaned.geojson`
- `data/processed/kc_house_data_cleaned.csv`

Reproduce with:

```powershell
cd project-3-geospatial-real-estate-valuation
python run_pipeline.py
python -m unittest tests.test_preprocessing -v
```
