import os
import sys

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.download_data import download_dataset
from src.data_preprocessing import preprocess_data

def main():
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    
    raw_csv_path = os.path.join(raw_data_dir, 'kc_house_data.csv')
    processed_geojson_path = os.path.join(processed_data_dir, 'kc_house_data_cleaned.geojson')
    processed_csv_path = os.path.join(processed_data_dir, 'kc_house_data_cleaned.csv')
    
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
    
    # 3. Print verification statistics
    print("\n--- Pipeline Verification Summary ---")
    print(f"Total rows in processed dataset: {gdf.shape[0]}")
    
    original_mean = gdf['price'].mean()
    normalized_mean = gdf['price_normalized'].mean()
    original_max = gdf['price'].max()
    normalized_max = gdf['price_normalized'].max()
    
    print(f"Original Price - Mean: ${original_mean:,.2f}, Max: ${original_max:,.2f}")
    print(f"Normalized Price - Mean: ${normalized_mean:,.2f}, Max: ${normalized_max:,.2f}")
    
    outlier_pct = (gdf['is_price_outlier'].sum() / len(gdf)) * 100
    print(f"Total extreme outliers capped: {gdf['is_price_outlier'].sum()} ({outlier_pct:.2f}%)")
    
    # Verify file existence
    print(f"\nChecking outputs:")
    print(f"  GeoJSON: {os.path.exists(processed_geojson_path)} ({os.path.getsize(processed_geojson_path) / 1024 / 1024:.2f} MB)")
    print(f"  CSV:     {os.path.exists(processed_csv_path)} ({os.path.getsize(processed_csv_path) / 1024 / 1024:.2f} MB)")
    print("\nVerification successful! Pipeline ran smoothly.")

if __name__ == "__main__":
    main()
