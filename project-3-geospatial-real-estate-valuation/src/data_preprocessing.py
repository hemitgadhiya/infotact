import os
import pandas as pd
import geopandas as gpd  # pyrefly: ignore
import numpy as np

def preprocess_data(raw_data_path, processed_data_path):
    """
    Load raw geospatial data, clean/preprocess it, and save the processed dataset.
    This includes:
    1. Parsing dates.
    2. Handling missing values and typing.
    3. Correcting known data entry issues (like the 33-bedroom outlier).
    4. Removing invalid/empty rows (e.g., bedrooms or bathrooms == 0, price <= 0).
    5. Constructing Point geometries and creating a GeoDataFrame.
    6. Local (ZIP code) outlier detection and price winsorization (normalization).
    7. Saving output as GeoJSON (for geospatial analysis) and CSV.
    """
    print(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    
    # 1. Parsing dates
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
    
    # 2. Typing and general cleaning
    df['id'] = df['id'].astype(str)
    df['zipcode'] = df['zipcode'].astype(str)
    
    # 3. Correct known typos / outliers
    # The famous 33 bedrooms house (id 2402100895) has 33 bedrooms for 1620 sqft.
    # It is widely recognized as a typo for 3 bedrooms.
    if (df['bedrooms'] == 33).any():
        print("Correcting known 33-bedroom outlier to 3 bedrooms...")
        df.loc[df['bedrooms'] == 33, 'bedrooms'] = 3
        
    # 4. Remove invalid entries (e.g. price <= 0, bedrooms == 0, bathrooms == 0)
    # Note: Some houses might legitimately have 0 bedrooms (e.g. small studios) but we'll 
    # keep them if the square footage is small, or drop them if they represent invalid records.
    # To be safe and clean, let's remove rows with 0 bedrooms or 0 bathrooms as they are likely
    # non-residential or data entry errors, which is standard for housing valuation models.
    initial_shape = df.shape
    df = df[df['price'] > 50000] # Remove extremely low prices (e.g., foreclosure/data errors)
    df = df[df['bedrooms'] > 0]
    df = df[df['bathrooms'] > 0]
    
    # Check coordinates are within reasonable King County bounding box
    # lat: [47.08, 47.78], long: [-122.54, -121.08] (buffered slightly to [47.10, 47.85] and [-122.60, -121.00])
    df = df[(df['lat'] >= 47.10) & (df['lat'] <= 47.85)]
    df = df[(df['long'] >= -122.60) & (df['long'] <= -121.00)]
    
    removed_rows = initial_shape[0] - df.shape[0]
    print(f"Removed {removed_rows} rows during basic cleaning (invalid values/out-of-bounds/low-prices).")
    
    # Deduplicate: If same property (id) is sold multiple times, keep the latest sale
    # This prevents data leakage and ensures a static snapshot.
    df = df.sort_values('date')
    df = df.drop_duplicates(subset=['id'], keep='last')
    print(f"Deduplicated dataset: kept latest sale per property. Final row count: {df.shape[0]}")
    
    # 5. Convert to GeoDataFrame
    print("Converting to GeoDataFrame (EPSG:4326)...")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['long'], df['lat']),
        crs="EPSG:4326"
    )
    
    # 6. Normalize extreme price outliers using GeoPandas
    # We will perform ZIP-code-based Winsorization (capping price at Q3 + 3.0 * IQR per zip code).
    # Since prices vary by location, a high price in Medina is normal, but an outlier in Auburn.
    print("Normalizing extreme price outliers per ZIP code...")
    
    def cap_price_by_zip(group):
        q1 = group['price'].quantile(0.25)
        q3 = group['price'].quantile(0.75)
        iqr = q3 - q1
        
        # Define upper cap as Q3 + 3.0 * IQR (standard extreme outlier threshold)
        upper_cap = q3 + 3.0 * iqr
        
        # Fallback if ZIP code has too few listings to compute valid IQR
        if pd.isna(upper_cap) or iqr == 0 or len(group) < 5:
            upper_cap = group['price'].quantile(0.99) # Fallback to 99th percentile
            
        # Ensure upper_cap is reasonable (no capping below $300k, to protect normal listings)
        upper_cap = max(upper_cap, 300000.0)
        
        group['price_upper_cap'] = upper_cap
        group['price_normalized'] = np.minimum(group['price'], upper_cap)
        group['is_price_outlier'] = group['price'] > upper_cap
        return group
        
    gdf = gdf.groupby('zipcode', group_keys=False).apply(cap_price_by_zip)
    
    outliers_count = gdf['is_price_outlier'].sum()
    print(f"Identified and capped {outliers_count} extreme price outliers across ZIP codes.")
    
    # 7. Save the outputs
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Save as GeoJSON
    geojson_path = processed_data_path
    if not geojson_path.endswith('.geojson'):
        geojson_path = os.path.splitext(processed_data_path)[0] + '.geojson'
        
    print(f"Saving GeoJSON to {geojson_path}...")
    # Convert dates to string representation before saving to GeoJSON as geojson driver 
    # handles datetime sometimes inconsistently depending on spatialite/gdal versions.
    gdf_to_save = gdf.copy()
    gdf_to_save['date'] = gdf_to_save['date'].dt.strftime('%Y-%m-%d')
    gdf_to_save.to_file(geojson_path, driver='GeoJSON')
    
    # Save as CSV (drop geometry column since lat/long are already columns)
    csv_path = os.path.splitext(geojson_path)[0] + '.csv'
    print(f"Saving CSV to {csv_path}...")
    df_to_save = pd.DataFrame(gdf_to_save.drop(columns='geometry'))
    df_to_save.to_csv(csv_path, index=False)
    
    print("Preprocessing complete!")
    return gdf
