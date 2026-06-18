"""Run baseline error analysis and export neighborhood limitation report."""

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from baseline_error_analysis import (  # noqa: E402
    compute_holdout_errors,
    identify_limitation_neighborhoods,
    summarize_errors_by_zipcode,
)
from feature_engineering import engineer_features  # noqa: E402


def load_engineered_features(input_path: str) -> pd.DataFrame:
    if input_path.endswith(".geojson"):
        import geopandas as gpd

        return gpd.read_file(input_path)

    df = pd.read_csv(input_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _format_zipcode(value) -> str:
    if pd.isna(value):
        return "unknown"
    return str(int(float(value)))


def build_markdown_report(
    zip_summary: pd.DataFrame,
    limitations: dict[str, pd.DataFrame],
    overall_mape: float,
) -> str:
    high_error = limitations["high_error_zipcodes"].head(5)
    gentrifying = limitations["gentrification_proxy_zipcodes"].head(5)
    luxury = limitations["luxury_market_zipcodes"].head(5)
    thresholds = limitations["thresholds"].iloc[0]

    lines = [
        "# XGBoost Baseline Limitations by Neighborhood",
        "",
        "This report summarizes where the tabular XGBoost baseline underperforms",
        "before spatial graph modeling is introduced in Week 3.",
        "",
        "## Overall holdout performance",
        "",
        f"- Overall holdout MAPE: **{overall_mape * 100:.2f}%**",
        f"- ZIP codes analyzed: **{len(zip_summary)}**",
        "",
        "## Why the baseline fails in some areas",
        "",
        "1. **Gentrification proxy areas** mix renovated homes with older neighbor comps.",
        "2. **Luxury ZIP codes** have heavy price tails that tabular features underfit.",
        "3. **Spatial spillovers** are missing because lat/long are dropped at training time.",
        "",
        "## Detection thresholds",
        "",
        f"- High-error MAPE cutoff: {thresholds['high_mape_threshold_pct']:.2f}%",
        f"- Gentrification MAPE cutoff: {thresholds['gentrification_mape_threshold_pct']:.2f}%",
        f"- Minimum listings per ZIP: {int(thresholds['min_listings'])}",
        "",
        "## Highest-error ZIP codes",
        "",
        "| ZIP | Listings | MAPE % | Mean Price | Renovation Rate |",
        "|-----|----------|--------|------------|-----------------|",
    ]

    for _, row in high_error.iterrows():
        lines.append(
            f"| {_format_zipcode(row['zipcode'])} | {int(row['listings'])} | {row['mape_pct']:.2f} | "
            f"${row['mean_actual_price']:,.0f} | {row['renovation_rate'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Gentrification-proxy ZIP codes",
            "",
            "| ZIP | Listings | MAPE % | Neighbor Ratio | Renovation Rate |",
            "|-----|----------|--------|----------------|-----------------|",
        ]
    )

    for _, row in gentrifying.iterrows():
        lines.append(
            f"| {_format_zipcode(row['zipcode'])} | {int(row['listings'])} | {row['mape_pct']:.2f} | "
            f"{row['mean_neighbor_ratio']:.2f} | {row['renovation_rate'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Luxury-market stress ZIP codes",
            "",
            "| ZIP | Listings | MAPE % | Mean Price |",
            "|-----|----------|--------|------------|",
        ]
    )

    for _, row in luxury.iterrows():
        lines.append(
            f"| {_format_zipcode(row['zipcode'])} | {int(row['listings'])} | {row['mape_pct']:.2f} | "
            f"${row['mean_actual_price']:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Implication for Week 3",
            "",
            "These neighborhoods justify graph/spatial embeddings because local neighbor",
            "context should explain price gaps that tabular features cannot capture alone.",
            "",
            "Regenerate this report with:",
            "",
            "```powershell",
            "python scripts/analyze_baseline_errors.py",
            "```",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline XGBoost errors by neighborhood.")
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "data", "processed", "kc_house_data_cleaned.csv"),
        help="Cleaned CSV or engineered CSV/GeoJSON input",
    )
    parser.add_argument(
        "--report",
        default=os.path.join(PROJECT_ROOT, "docs", "baseline_limitations_by_neighborhood.md"),
        help="Markdown report output path",
    )
    args = parser.parse_args()

    features_df = load_engineered_features(args.input)
    if "house_age" not in features_df.columns:
        features_df = engineer_features(features_df)

    error_df = compute_holdout_errors(features_df)
    zip_summary = summarize_errors_by_zipcode(error_df)
    limitations = identify_limitation_neighborhoods(zip_summary)
    overall_mape = error_df["percentage_error"].mean()

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    report = build_markdown_report(zip_summary, limitations, overall_mape)
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"Overall holdout MAPE: {overall_mape * 100:.2f}%")
    print(f"High-error ZIP codes: {len(limitations['high_error_zipcodes'])}")
    print(
        "Gentrification-proxy ZIP codes: "
        f"{len(limitations['gentrification_proxy_zipcodes'])}"
    )
    print(f"Report saved to: {args.report}")


if __name__ == "__main__":
    main()
