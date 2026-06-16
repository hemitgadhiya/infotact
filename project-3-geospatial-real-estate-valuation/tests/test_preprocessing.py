import os
import unittest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from src.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_raw_path = os.path.join(self.test_dir, "temp_raw.csv")
        self.temp_processed_path = os.path.join(self.test_dir, "temp_processed.geojson")
        self.temp_csv_path = os.path.join(self.test_dir, "temp_processed.csv")
        
        # Create a mock dataset representing various edge cases:
        # 1. Normal record
        # 2. typo: 33 bedrooms
        # 3. duplicate transactions (same id)
        # 4. invalid low price
        # 5. out of bounds lat/long coordinate
        # 6. missing bathrooms (0.0)
        # 7. extreme price outlier
        data = {
            'id': [1, 2, 3, 3, 4, 5, 6, 7, 8, 9],
            'date': [
                '20141013T000000', '20141013T000000', 
                '20141013T000000', '20151013T000000', # id 3 sold twice
                '20141013T000000', '20141013T000000',
                '20141013T000000', '20141013T000000',
                '20141013T000000', '20141013T000000'
            ],
            'price': [200000, 250000, 220000, 210000, 230000, 10000000, 350000, 400000, 0, 450000], # 8 is price 0
            'bedrooms': [3, 33, 3, 3, 3, 4, 3, 3, 3, 3], # 2 is 33 bedrooms typo
            'bathrooms': [2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.0, 2.0, 2.0, 0.0], # 9 is 0 bathrooms
            'sqft_living': [1500, 1600, 1500, 1500, 1500, 3000, 1800, 2000, 1500, 1500],
            'sqft_lot': [5000, 5000, 5000, 5000, 5000, 8000, 6000, 7000, 5000, 5000],
            'floors': [1, 1, 1, 1, 1, 2, 1, 1.5, 1, 1],
            'waterfront': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'view': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'condition': [3, 3, 3, 3, 3, 4, 3, 3, 3, 3],
            'grade': [7, 7, 7, 7, 7, 8, 7, 8, 7, 7],
            'sqft_above': [1500, 1600, 1500, 1500, 1500, 3000, 1800, 2000, 1500, 1500],
            'sqft_basement': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'yr_built': [1990, 1990, 1990, 1990, 1990, 2005, 1995, 1998, 1990, 1990],
            'yr_renovated': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'zipcode': [98001, 98001, 98001, 98001, 98001, 98001, 98001, 98001, 98001, 98001],
            'lat': [47.3, 47.3, 47.3, 47.3, 47.3, 47.3, 47.3, 10.0, 47.3, 47.3], # 7 is out of bounds
            'long': [-122.2, -122.2, -122.2, -122.2, -122.2, -122.2, -122.2, -122.2, -122.2, -122.2],
            'sqft_living15': [1500, 1600, 1500, 1500, 1500, 3000, 1800, 2000, 1500, 1500],
            'sqft_lot15': [5000, 5000, 5000, 5000, 5000, 8000, 6000, 7000, 5000, 5000]
        }
        pd.DataFrame(data).to_csv(self.temp_raw_path, index=False)
        
    def tearDown(self):
        # Clean up files if they exist
        for f in [self.temp_raw_path, self.temp_processed_path, self.temp_csv_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
                
    def test_preprocessing(self):
        gdf = preprocess_data(self.temp_raw_path, self.temp_processed_path)
        
        # Verify columns and geometry
        self.assertIn('geometry', gdf.columns)
        self.assertTrue(isinstance(gdf, gpd.GeoDataFrame))
        
        # Verify invalid rows filtered out:
        # id 7 (out-of-bounds lat) -> filtered
        # id 8 (price = 0) -> filtered
        # id 9 (bathrooms = 0) -> filtered
        self.assertNotIn('7', gdf['id'].values)
        self.assertNotIn('8', gdf['id'].values)
        self.assertNotIn('9', gdf['id'].values)
        
        # Verify deduplication:
        # id 3 had two sales. The kept transaction should be the latest (price 210000).
        id_3_rows = gdf[gdf['id'] == '3']
        self.assertEqual(len(id_3_rows), 1)
        self.assertEqual(id_3_rows.iloc[0]['price'], 210000)
        
        # Verify 33-bedroom correction:
        # id 2 had 33 bedrooms, should be corrected to 3
        id_2_rows = gdf[gdf['id'] == '2']
        self.assertEqual(id_2_rows.iloc[0]['bedrooms'], 3)
        
        # Verify price winsorization/capping:
        # id 5 had price 10,000,000. It should be capped.
        id_5_rows = gdf[gdf['id'] == '5']
        self.assertTrue(id_5_rows.iloc[0]['is_price_outlier'])
        self.assertLess(id_5_rows.iloc[0]['price_normalized'], 10000000.0)
        
        # Verify output files exist
        self.assertTrue(os.path.exists(self.temp_processed_path))
        self.assertTrue(os.path.exists(self.temp_csv_path))

if __name__ == '__main__':
    unittest.main()
