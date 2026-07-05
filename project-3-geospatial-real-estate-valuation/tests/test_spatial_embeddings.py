import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import joblib
from src.spatial_embeddings import compute_neighborhood_features, generate_spatial_embeddings


class TestSpatialEmbeddings(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset representing 4 properties
        # properties: house_A, house_B, house_C, house_D
        self.df = pd.DataFrame(
            {
                "id": ["house_A", "house_B", "house_C", "house_D"],
                "price_normalized": [10.0, 20.0, 30.0, 40.0],
                "sqft_living": [1000.0, 2000.0, 3000.0, 4000.0],
                "house_age": [5.0, 15.0, 25.0, 35.0],
            }
        )

        # Mock KNN adjacency list
        # format: source_id -> list of (neighbor_id, distance_km)
        # house_A is close to B (1.0km) and C (2.0km)
        # house_B is close to A (1.0km) and C (1.0km)
        # house_C is close to B (1.0km) and D (2.0km)
        # house_D is close to C (2.0km)
        self.adj = {
            "house_A": [("house_B", 1.0), ("house_C", 2.0)],
            "house_B": [("house_A", 1.0), ("house_C", 1.0)],
            "house_C": [("house_B", 1.0), ("house_D", 2.0)],
            "house_D": [("house_C", 2.0)],
        }

        # Mock edges DataFrame representation
        self.edges_df = pd.DataFrame(
            [
                {"source_id": "house_A", "target_id": "house_B", "distance_km": 1.0},
                {"source_id": "house_A", "target_id": "house_C", "distance_km": 2.0},
                {"source_id": "house_B", "target_id": "house_A", "distance_km": 1.0},
                {"source_id": "house_B", "target_id": "house_C", "distance_km": 1.0},
                {"source_id": "house_C", "target_id": "house_B", "distance_km": 1.0},
                {"source_id": "house_C", "target_id": "house_D", "distance_km": 2.0},
                {"source_id": "house_D", "target_id": "house_C", "distance_km": 2.0},
            ]
        )

        # Temporary directory for saving scaler/PCA artifacts
        self.temp_dir = tempfile.mkdtemp()
        self.scaler_path = os.path.join(self.temp_dir, "scaler.pkl")
        self.pca_path = os.path.join(self.temp_dir, "pca.pkl")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_compute_neighborhood_features(self):
        res = compute_neighborhood_features(self.df, self.adj, id_col="id")

        # 4 properties expected
        self.assertEqual(len(res), 4)

        # Check values for house_A: neighbors B (price 20, sqft 2000, age 15, dist 1) and C (price 30, sqft 3000, age 25, dist 2)
        row_a = res[res["id"] == "house_A"].iloc[0]
        self.assertEqual(row_a["local_price_mean"], 25.0)  # (20 + 30) / 2
        self.assertEqual(row_a["local_price_median"], 25.0)
        self.assertEqual(row_a["local_price_min"], 20.0)
        self.assertEqual(row_a["local_price_max"], 30.0)
        self.assertAlmostEqual(row_a["local_price_std"], 5.0)  # std of [20, 30] is 5.0
        self.assertEqual(row_a["local_sqft_living_mean"], 2500.0)  # (2000 + 3000) / 2
        self.assertEqual(row_a["local_house_age_mean"], 20.0)  # (15 + 25) / 2
        self.assertEqual(row_a["local_distance_mean"], 1.5)  # (1 + 2) / 2

        # Weighted mean calculation for house_A:
        # w_B = 1 / (1.0 + 0.01) = 1 / 1.01 approx 0.990099
        # w_C = 1 / (2.0 + 0.01) = 1 / 2.01 approx 0.497512
        # weighted_price_sum = (20 * w_B) + (30 * w_C)
        # weighted_mean = weighted_price_sum / (w_B + w_C)
        w_b = 1.0 / 1.01
        w_c = 1.0 / 2.01
        expected_weighted_mean = ((20.0 * w_b) + (30.0 * w_c)) / (w_b + w_c)
        self.assertAlmostEqual(row_a["local_price_weighted_mean"], expected_weighted_mean, places=5)

        # Price per sqft for B: 20 / 2000 = 0.01. For C: 30 / 3000 = 0.01. Mean = 0.01.
        self.assertAlmostEqual(row_a["local_price_per_sqft_mean"], 0.01, places=5)

    def test_missing_neighbors_and_lookups(self):
        # Create a graph with an isolated node and a node pointing to a non-existent ID
        incomplete_adj = {
            "house_A": [],  # no neighbors
            "house_B": [("non_existent_house", 1.0)],  # missing neighbor in df
        }
        res = compute_neighborhood_features(self.df, incomplete_adj, id_col="id")

        row_a = res[res["id"] == "house_A"].iloc[0]
        row_b = res[res["id"] == "house_B"].iloc[0]

        # Verify fallback to 0.0
        self.assertEqual(row_a["local_price_mean"], 0.0)
        self.assertEqual(row_a["local_price_weighted_mean"], 0.0)
        self.assertEqual(row_b["local_price_mean"], 0.0)

    def test_generate_spatial_embeddings_shapes_and_pca(self):
        # Generate with n_components = 2
        emb_df, raw_df = generate_spatial_embeddings(
            self.df,
            self.edges_df,
            n_components=2,
            id_col="id",
            scaler_path=self.scaler_path,
            pca_path=self.pca_path,
        )

        # Check shapes
        self.assertEqual(len(emb_df), 4)
        self.assertEqual(len(raw_df), 4)
        self.assertIn("spatial_emb_0", emb_df.columns)
        self.assertIn("spatial_emb_1", emb_df.columns)
        self.assertNotIn("spatial_emb_2", emb_df.columns)

        # Check saved scaler and pca exist on disk
        self.assertTrue(os.path.exists(self.scaler_path))
        self.assertTrue(os.path.exists(self.pca_path))

        # Check reload reproducibility
        emb_df_reloaded, _ = generate_spatial_embeddings(
            self.df,
            self.edges_df,
            n_components=2,
            id_col="id",
            scaler_path=self.scaler_path,
            pca_path=self.pca_path,
        )

        pd.testing.assert_frame_equal(emb_df, emb_df_reloaded)


if __name__ == "__main__":
    unittest.main()
