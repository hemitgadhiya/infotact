import os
import sys
import pandas as pd  # pyrefly: ignore

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.download_data import download_dataset  # pyrefly: ignore
from src.data_preprocessing import preprocess_data  # pyrefly: ignore
from src.feature_engineering import engineer_features  # pyrefly: ignore

def main():
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    
    raw_csv_path = os.path.join(raw_data_dir, 'kc_house_data.csv')
    processed_geojson_path = os.path.join(processed_data_dir, 'kc_house_data_cleaned.geojson')
    processed_csv_path = os.path.join(processed_data_dir, 'kc_house_data_cleaned.csv')
    
    engineered_geojson_path = os.path.join(processed_data_dir, 'kc_house_data_engineered.geojson')
    engineered_csv_path = os.path.join(processed_data_dir, 'kc_house_data_engineered.csv')
    
    # 1. Download data if it doesn't exist
    url = "https://raw.githubusercontent.com/jmatth11/King-County-House-Data-Set/master/kc_house_data.csv"
    if not os.path.exists(raw_csv_path):
        print("Raw dataset not found locally. Starting download...")
        download_dataset(url, raw_csv_path)
    else:
        print(f"Raw dataset already exists at {raw_csv_path}")
        
    # 2. Run preprocessing
    print("\nStarting preprocessing pipeline...")
    gdf = preprocess_data(raw_csv_path, processed_geojson_path)
    
    # 3. Run feature engineering
    print("\nStarting feature engineering pipeline...")
    gdf_engineered = engineer_features(gdf)
    
    print(f"Saving engineered GeoJSON to {engineered_geojson_path}...")
    # Convert dates to string representation before saving to GeoJSON
    gdf_eng_to_save = gdf_engineered.copy()
    gdf_eng_to_save['date'] = gdf_eng_to_save['date'].dt.strftime('%Y-%m-%d')
    gdf_eng_to_save.to_file(engineered_geojson_path, driver='GeoJSON')
    
    print(f"Saving engineered CSV to {engineered_csv_path}...")
    df_eng_to_save = pd.DataFrame(gdf_eng_to_save.drop(columns='geometry'))
    df_eng_to_save.to_csv(engineered_csv_path, index=False)
    
    # 4. Print verification statistics
    print("\n--- Pipeline Verification Summary ---")
    print(f"Total rows in processed dataset: {gdf_engineered.shape[0]}")
    
    original_mean = gdf_engineered['price'].mean()
    normalized_mean = gdf_engineered['price_normalized'].mean()
    original_max = gdf_engineered['price'].max()
    normalized_max = gdf_engineered['price_normalized'].max()
    
    print(f"Original Price - Mean: ${original_mean:,.2f}, Max: ${original_max:,.2f}")
    print(f"Normalized Price - Mean: ${normalized_mean:,.2f}, Max: ${normalized_max:,.2f}")
    
    outlier_pct = (gdf_engineered['is_price_outlier'].sum() / len(gdf_engineered)) * 100
    print(f"Total extreme outliers capped: {gdf_engineered['is_price_outlier'].sum()} ({outlier_pct:.2f}%)")
    
    # Print some statistics about engineered features
    mean_age = gdf_engineered['house_age'].mean()
    mean_dist_seattle = gdf_engineered['dist_to_seattle_center_km'].mean()
    mean_dist_bellevue = gdf_engineered['dist_to_bellevue_center_km'].mean()
    renovated_pct = (gdf_engineered['is_renovated'].sum() / len(gdf_engineered)) * 100
    
    print(f"Mean House Age: {mean_age:.1f} years")
    print(f"Renovated properties: {gdf_engineered['is_renovated'].sum()} ({renovated_pct:.2f}%)")
    print(f"Mean Distance to Seattle Center: {mean_dist_seattle:.2f} km")
    print(f"Mean Distance to Bellevue Center: {mean_dist_bellevue:.2f} km")
    
    # Verify file existence
    print(f"\nChecking outputs:")
    print(f"  Cleaned GeoJSON:    {os.path.exists(processed_geojson_path)} ({os.path.getsize(processed_geojson_path) / 1024 / 1024:.2f} MB)")
    print(f"  Cleaned CSV:        {os.path.exists(processed_csv_path)} ({os.path.getsize(processed_csv_path) / 1024 / 1024:.2f} MB)")
    print(f"  Engineered GeoJSON: {os.path.exists(engineered_geojson_path)} ({os.path.getsize(engineered_geojson_path) / 1024 / 1024:.2f} MB)")
    print(f"  Engineered CSV:     {os.path.exists(engineered_csv_path)} ({os.path.getsize(engineered_csv_path) / 1024 / 1024:.2f} MB)")
    print("\nVerification successful! Pipeline ran smoothly.")

if __name__ == "__main__":
    main()
