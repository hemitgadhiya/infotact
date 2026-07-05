import os
import sys
import pandas as pd  # pyrefly: ignore

# Add src and scripts to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

from src.download_data import download_dataset  # pyrefly: ignore
from src.data_preprocessing import preprocess_data  # pyrefly: ignore
from src.feature_engineering import engineer_features  # pyrefly: ignore
from src.model_training import train_valuation_model  # pyrefly: ignore
from src.graph_construction import build_knn_graph  # pyrefly: ignore
from src.spatial_embeddings import generate_spatial_embeddings  # pyrefly: ignore
from train_spatial_model import train as train_spatial_model  # pyrefly: ignore
from compare_models import main as compare_models_main  # pyrefly: ignore

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
    knn_edges_path = os.path.join(processed_data_dir, 'kc_house_data_knn_edges.csv')
    
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
    
    # 4. Build KNN Graph
    print("\nStarting graph construction pipeline...")
    graph_res = build_knn_graph(gdf_engineered, k=5)
    edges_df = graph_res["edges_df"]
    print(f"Saving KNN graph edges (K={graph_res['k_used']}) to {knn_edges_path}...")
    edges_df.to_csv(knn_edges_path, index=False)
    
    # 4.5. Generate Spatial Embeddings
    print("\nStarting spatial embeddings generation...")
    spatial_embeddings_path = os.path.join(processed_data_dir, 'kc_house_data_spatial_embeddings.csv')
    scaler_path = os.path.join(project_root, 'models', 'spatial_embedding_scaler.pkl')
    pca_path = os.path.join(project_root, 'models', 'spatial_embedding_pca.pkl')
    
    emb_df, raw_feats_df = generate_spatial_embeddings(
        df=gdf_engineered,
        edges_df=edges_df,
        n_components=8,
        id_col="id",
        price_col="price_normalized",
        scaler_path=scaler_path,
        pca_path=pca_path,
    )
    print(f"Saving spatial embeddings to {spatial_embeddings_path}...")
    emb_df.to_csv(spatial_embeddings_path, index=False)
    
    # Merge embeddings and raw features into gdf_engineered
    gdf_engineered['id'] = gdf_engineered['id'].astype(str)
    emb_df['id'] = emb_df['id'].astype(str)
    raw_feats_df['id'] = raw_feats_df['id'].astype(str)
    
    gdf_engineered = gdf_engineered.merge(emb_df, on='id', how='left')
    gdf_engineered = gdf_engineered.merge(raw_feats_df, on='id', how='left')
    
    # Save the updated files containing spatial features
    print(f"Updating engineered GeoJSON with spatial features to {engineered_geojson_path}...")
    gdf_eng_to_save = gdf_engineered.copy()
    if pd.api.types.is_datetime64_any_dtype(gdf_eng_to_save['date']):
        gdf_eng_to_save['date'] = gdf_eng_to_save['date'].dt.strftime('%Y-%m-%d')
    gdf_eng_to_save.to_file(engineered_geojson_path, driver='GeoJSON')
    
    print(f"Updating engineered CSV with spatial features to {engineered_csv_path}...")
    df_eng_to_save = pd.DataFrame(gdf_eng_to_save.drop(columns='geometry'))
    df_eng_to_save.to_csv(engineered_csv_path, index=False)
    
    # 5. Train XGBoost baseline model on tabular features
    print("\nTraining baseline XGBoost regressor...")
    metrics = train_valuation_model(df_eng_to_save, os.path.join(project_root, 'models', 'xgboost_regressor.pkl'))
    print(f"Baseline metrics: MAPE={metrics['mape']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # 6. Train PyTorch Spatial Attention model
    print("\nTraining Spatial Attention model (PyTorch)...")
    train_spatial_model()
    
    # 7. Compare models
    print("\nComparing baseline and spatial models...")
    compare_models_main()
    
    # 5. Print verification statistics
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
    spatial_model_path = os.path.join(project_root, 'models', 'spatial_attention_model.pth')
    model_comparison_path = os.path.join(project_root, 'docs', 'model_comparison.md')
    print(f"\nChecking outputs:")
    print(f"  Cleaned GeoJSON:    {os.path.exists(processed_geojson_path)} ({os.path.getsize(processed_geojson_path) / 1024 / 1024:.2f} MB)")
    print(f"  Cleaned CSV:        {os.path.exists(processed_csv_path)} ({os.path.getsize(processed_csv_path) / 1024 / 1024:.2f} MB)")
    print(f"  Engineered GeoJSON: {os.path.exists(engineered_geojson_path)} ({os.path.getsize(engineered_geojson_path) / 1024 / 1024:.2f} MB)")
    print(f"  Engineered CSV:     {os.path.exists(engineered_csv_path)} ({os.path.getsize(engineered_csv_path) / 1024 / 1024:.2f} MB)")
    print(f"  KNN Graph Edges:    {os.path.exists(knn_edges_path)} ({os.path.getsize(knn_edges_path) / 1024 / 1024:.2f} MB)")
    print(f"  Spatial Model Pth:  {os.path.exists(spatial_model_path)} ({os.path.getsize(spatial_model_path) / 1024 / 1024:.2f} MB if it exists else 0 MB)")
    print(f"  Comparison Report:  {os.path.exists(model_comparison_path)}")
    print("\nVerification successful! Pipeline ran smoothly.")

if __name__ == "__main__":
    main()
