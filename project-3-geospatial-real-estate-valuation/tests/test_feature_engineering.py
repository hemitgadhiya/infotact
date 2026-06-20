import unittest
import pandas as pd  # pyrefly: ignore
import geopandas as gpd  # pyrefly: ignore
import numpy as np  # pyrefly: ignore
from shapely.geometry import Point  # pyrefly: ignore
from src.feature_engineering import engineer_features, haversine_distance  # pyrefly: ignore

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Create mock data with 3 rows
        self.mock_data = pd.DataFrame({
            'id': ['1', '2', '3'],
            'date': [
                '2014-10-13 00:00:00',
                '2015-05-15 00:00:00',
                '2014-12-01 00:00:00'
            ],
            'bedrooms': [3, 4, 2],
            'bathrooms': [2.0, 2.5, 1.0],
            'sqft_living': [1500, 3000, 1000],
            'sqft_lot': [5000, 8000, 3000],
            'sqft_basement': [0, 1000, 0],
            'yr_built': [1990, 2005, 1950],
            'yr_renovated': [0, 2010, 0],
            'lat': [47.6062, 47.6101, 47.5],
            'long': [-122.3321, -122.2015, -122.4],
            'sqft_living15': [1600, 2800, 1100],
            'sqft_lot15': [4800, 7500, 3200]
        })
        
        self.mock_gdf = gpd.GeoDataFrame(
            self.mock_data,
            geometry=gpd.points_from_xy(self.mock_data['long'], self.mock_data['lat']),
            crs="EPSG:4326"
        )
        
    def test_haversine_distance(self):
        # Distance between Seattle and Bellevue centers should be ~9.8 km
        dist = haversine_distance(47.6062, -122.3321, 47.6101, -122.2015)
        self.assertAlmostEqual(dist, 9.81, places=1)
        
    def test_engineer_features(self):
        df_feat = engineer_features(self.mock_data)
        
        # Verify columns exist
        expected_cols = [
            'house_age', 'is_renovated', 'years_since_renovation',
            'total_rooms', 'bedrooms_per_bathroom', 'has_basement',
            'sqft_living_to_lot_ratio', 'sqft_living_per_bedroom',
            'sqft_living_per_bathroom', 'sqft_living_vs_neighbor_ratio',
            'sqft_lot_vs_neighbor_ratio', 'dist_to_seattle_center_km',
            'dist_to_bellevue_center_km', 'sale_year', 'sale_month', 'sale_quarter'
        ]
        for col in expected_cols:
            self.assertIn(col, df_feat.columns)
            
        # Verify age calculations
        # Row 1: built 1990, sale 2014 -> age 24
        # Row 2: built 2005, sale 2015 -> age 10
        self.assertEqual(df_feat.iloc[0]['house_age'], 24)
        self.assertEqual(df_feat.iloc[1]['house_age'], 10)
        
        # Verify renovation logic
        # Row 1: yr_renovated is 0 -> is_renovated=0, years_since_renovation=24
        # Row 2: yr_renovated is 2010 -> is_renovated=1, years_since_renovation=5 (2015 - 2010)
        self.assertEqual(df_feat.iloc[0]['is_renovated'], 0)
        self.assertEqual(df_feat.iloc[1]['is_renovated'], 1)
        self.assertEqual(df_feat.iloc[0]['years_since_renovation'], 24)
        self.assertEqual(df_feat.iloc[1]['years_since_renovation'], 5)
        
        # Verify room ratios
        # Row 1: bedrooms=3, bathrooms=2.0 -> ratio=1.5, total=5.0
        self.assertEqual(df_feat.iloc[0]['total_rooms'], 5.0)
        self.assertEqual(df_feat.iloc[0]['bedrooms_per_bathroom'], 1.5)
        
        # Verify basement indicator
        # Row 1: sqft_basement=0 -> has_basement=0
        # Row 2: sqft_basement=1000 -> has_basement=1
        self.assertEqual(df_feat.iloc[0]['has_basement'], 0)
        self.assertEqual(df_feat.iloc[1]['has_basement'], 1)
        
        # Verify distance calculations
        # Row 1: Seattle center coordinates -> dist to Seattle center should be 0
        self.assertAlmostEqual(df_feat.iloc[0]['dist_to_seattle_center_km'], 0.0, places=3)
        # Row 2: Bellevue center coordinates -> dist to Seattle center should be ~9.8 km
        self.assertAlmostEqual(df_feat.iloc[1]['dist_to_seattle_center_km'], 9.81, places=1)
        # Row 2: dist to Bellevue center should be 0
        self.assertAlmostEqual(df_feat.iloc[1]['dist_to_bellevue_center_km'], 0.0, places=3)
        
        # Verify seasonality
        # Row 1: '2014-10-13 00:00:00' -> year=2014, month=10, quarter=4
        self.assertEqual(df_feat.iloc[0]['sale_year'], 2014)
        self.assertEqual(df_feat.iloc[0]['sale_month'], 10)
        self.assertEqual(df_feat.iloc[0]['sale_quarter'], 4)

    def test_engineer_features_edge_cases(self):
        # Create mock data with 0 bedrooms and 0 bathrooms to check fallback logic
        edge_data = pd.DataFrame({
            'id': ['4'],
            'date': ['2015-01-01 00:00:00'],
            'bedrooms': [0],
            'bathrooms': [0.0],
            'sqft_living': [1000],
            'sqft_lot': [3000],
            'sqft_basement': [0],
            'yr_built': [1950],
            'yr_renovated': [0],
            'lat': [47.5],
            'long': [-122.4],
            'sqft_living15': [1100],
            'sqft_lot15': [3200]
        })
        df_feat = engineer_features(edge_data)
        
        # Verify bedrooms_per_bathroom fallback to bedrooms (0)
        self.assertEqual(df_feat.iloc[0]['bedrooms_per_bathroom'], 0)
        # Verify sqft_living_per_bedroom fallback to sqft_living (1000)
        self.assertEqual(df_feat.iloc[0]['sqft_living_per_bedroom'], 1000)
        # Verify sqft_living_per_bathroom fallback to sqft_living (1000)
        self.assertEqual(df_feat.iloc[0]['sqft_living_per_bathroom'], 1000)

    def test_geodataframe_compatibility(self):
        # Test that passing a GeoDataFrame keeps the geometry and returns a GeoDataFrame
        gdf_feat = engineer_features(self.mock_gdf)
        self.assertTrue(isinstance(gdf_feat, gpd.GeoDataFrame))
        self.assertIn('geometry', gdf_feat.columns)
        self.assertEqual(gdf_feat.crs, "EPSG:4326")

if __name__ == '__main__':
    unittest.main()
