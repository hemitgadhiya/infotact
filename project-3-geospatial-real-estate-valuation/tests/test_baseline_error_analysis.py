import unittest

import pandas as pd

from src.baseline_error_analysis import (
    identify_limitation_neighborhoods,
    summarize_errors_by_zipcode,
)


class TestBaselineErrorAnalysis(unittest.TestCase):
    def setUp(self):
        self.error_df = pd.DataFrame(
            {
                "zipcode": ["98101", "98101", "98102", "98102", "98103", "98103"],
                "actual_price": [500000, 600000, 450000, 470000, 900000, 950000],
                "predicted_price": [480000, 540000, 430000, 520000, 700000, 720000],
                "absolute_error": [20000, 60000, 20000, 50000, 200000, 230000],
                "percentage_error": [0.04, 0.10, 0.044, 0.106, 0.222, 0.242],
                "sqft_living_vs_neighbor_ratio": [1.1, 1.2, 1.4, 1.5, 1.0, 1.05],
                "is_renovated": [0, 1, 1, 1, 0, 0],
            }
        )

        self.zip_summary = summarize_errors_by_zipcode(self.error_df)

    def test_summarize_errors_by_zipcode(self):
        self.assertEqual(len(self.zip_summary), 3)
        self.assertIn("mape_pct", self.zip_summary.columns)
        self.assertGreater(self.zip_summary.iloc[0]["mape"], 0)

    def test_identify_limitation_neighborhoods(self):
        limitations = identify_limitation_neighborhoods(
            self.zip_summary,
            high_mape_quantile=0.67,
            gentrification_mape_quantile=0.50,
            min_listings=2,
        )

        self.assertIn("high_error_zipcodes", limitations)
        self.assertIn("gentrification_proxy_zipcodes", limitations)
        self.assertFalse(limitations["high_error_zipcodes"].empty)
        self.assertFalse(limitations["gentrification_proxy_zipcodes"].empty)


if __name__ == "__main__":
    unittest.main()
