import unittest
import numpy as np
import pandas as pd
from src.graph_construction import build_knn_graph  # pyrefly: ignore
from src.feature_engineering import haversine_distance  # pyrefly: ignore


class TestGraphConstruction(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset with 4 rows representing distinct locations
        # Row 0 and Row 1 are Seattle-ish coordinates (~9.8 km apart)
        # Row 2 and Row 3 are other locations
        self.mock_df = pd.DataFrame(
            {
                "id": ["house_A", "house_B", "house_C", "house_D"],
                "lat": [47.6062, 47.6101, 47.5000, 47.8000],
                "long": [-122.3321, -122.2015, -122.4000, -122.1000],
            }
        )

    def test_basic_graph_construction(self):
        # Test constructing a graph with K=2
        res = build_knn_graph(self.mock_df, k=2)

        edges_df = res["edges_df"]
        adj_list = res["adjacency_list"]
        k_used = res["k_used"]

        # Validate structure and shapes
        self.assertEqual(k_used, 2)
        # Since K=2 and we have 4 nodes, each node must have exactly 2 edges -> 8 edges total
        self.assertEqual(len(edges_df), 8)
        self.assertEqual(len(adj_list), 4)

        # Check that each node is represented in edges
        self.assertEqual(set(edges_df["source_id"].unique()), {"house_A", "house_B", "house_C", "house_D"})
        for node_id in self.mock_df["id"]:
            self.assertEqual(len(adj_list[node_id]), 2)

    def test_distance_accuracy_and_self_loops(self):
        # Test K=1
        res = build_knn_graph(self.mock_df, k=1)
        edges_df = res["edges_df"]
        adj_list = res["adjacency_list"]

        # The closest neighbor to house_A (Seattle Center) should be house_B (Bellevue Center)
        # Verify distance matches custom haversine calculation
        house_a_edges = edges_df[edges_df["source_id"] == "house_A"]
        self.assertEqual(len(house_a_edges), 1)

        target_id = house_a_edges.iloc[0]["target_id"]
        dist_km = house_a_edges.iloc[0]["distance_km"]

        self.assertEqual(target_id, "house_B")

        expected_dist = haversine_distance(47.6062, -122.3321, 47.6101, -122.2015)
        self.assertAlmostEqual(dist_km, expected_dist, places=3)

        # Ensure no self loops exist: no node connects to itself
        for _, row in edges_df.iterrows():
            self.assertNotEqual(row["source_id"], row["target_id"])

        for src, targets in adj_list.items():
            for dest, _ in targets:
                self.assertNotEqual(src, dest)

    def test_duplicate_coordinates(self):
        # Create a dataframe where house_A and house_B have the exact same lat/long
        dup_df = pd.DataFrame(
            {
                "id": ["house_A", "house_B", "house_C"],
                "lat": [47.6062, 47.6062, 47.5000],
                "long": [-122.3321, -122.3321, -122.4000],
            }
        )

        res = build_knn_graph(dup_df, k=1)
        edges_df = res["edges_df"]
        adj_list = res["adjacency_list"]

        # house_A and house_B are identical, so distance between them is 0
        # house_A's nearest neighbor should be house_B with distance 0
        house_a_edges = edges_df[edges_df["source_id"] == "house_A"]
        self.assertEqual(house_a_edges.iloc[0]["target_id"], "house_B")
        self.assertAlmostEqual(house_a_edges.iloc[0]["distance_km"], 0.0, places=5)

        # Ensure house_A did not self-loop to house_A
        self.assertNotEqual(house_a_edges.iloc[0]["target_id"], "house_A")

    def test_small_datasets(self):
        # Dataset size N=2. If we request K=5, it must be capped to N-1 = 1.
        small_df = pd.DataFrame(
            {
                "id": ["house_A", "house_B"],
                "lat": [47.6062, 47.6101],
                "long": [-122.3321, -122.2015],
            }
        )

        res = build_knn_graph(small_df, k=5)
        self.assertEqual(res["k_used"], 1)
        self.assertEqual(len(res["edges_df"]), 2)  # A->B and B->A

        # Dataset size N=1. Max possible K is 0.
        single_df = pd.DataFrame(
            {
                "id": ["house_A"],
                "lat": [47.6062],
                "long": [-122.3321],
            }
        )
        res_single = build_knn_graph(single_df, k=5)
        self.assertEqual(res_single["k_used"], 0)
        self.assertEqual(len(res_single["edges_df"]), 0)
        self.assertEqual(res_single["adjacency_list"], {"house_A": []})

    def test_missing_or_custom_columns(self):
        # Missing identifier column: should log warning and fallback to index
        no_id_df = pd.DataFrame(
            {
                "lat": [47.6062, 47.6101],
                "long": [-122.3321, -122.2015],
            }
        )
        res = build_knn_graph(no_id_df, k=1)
        self.assertEqual(res["k_used"], 1)
        self.assertEqual(set(res["edges_df"]["source_id"].unique()), {"0", "1"})

        # Missing coordinate columns: should raise KeyError
        with self.assertRaises(KeyError):
            build_knn_graph(pd.DataFrame({"id": ["A"], "lat": [47.0]}), k=1)

        # Custom coordinate column names
        custom_col_df = pd.DataFrame(
            {
                "id": ["A", "B"],
                "latitude": [47.6062, 47.6101],
                "longitude": [-122.3321, -122.2015],
            }
        )
        res_custom = build_knn_graph(
            custom_col_df, k=1, lat_col="latitude", long_col="longitude"
        )
        self.assertEqual(res_custom["k_used"], 1)
        self.assertEqual(len(res_custom["edges_df"]), 2)


if __name__ == "__main__":
    unittest.main()
