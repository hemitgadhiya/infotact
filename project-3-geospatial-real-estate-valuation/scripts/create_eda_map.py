"""Generate interactive Folium maps for geospatial price exploration."""

import argparse
import os
import sys

import folium
from folium.plugins import HeatMap
import geopandas as gpd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

DEFAULT_INPUT = os.path.join(
    PROJECT_ROOT, "data", "processed", "kc_house_data_cleaned.geojson"
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notebooks", "outputs")


def load_processed_data(input_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {input_path}. Run run_pipeline.py first."
        )
    return gpd.read_file(input_path)


def create_price_scatter_map(gdf: gpd.GeoDataFrame, output_path: str) -> str:
    """Plot property locations colored by normalized price."""
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()

    price_map = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")

    sample = gdf.sample(n=min(2000, len(gdf)), random_state=42)
    for _, row in sample.iterrows():
        price = row.get("price_normalized", row["price"])
        color = "red" if price >= 1_000_000 else "blue" if price >= 500_000 else "green"
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"Price: ${price:,.0f}<br>Bedrooms: {row['bedrooms']}",
        ).add_to(price_map)

    folium.LayerControl().add_to(price_map)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    price_map.save(output_path)
    return output_path


def create_price_heatmap(gdf: gpd.GeoDataFrame, output_path: str) -> str:
    """Plot a heatmap of normalized sale prices across King County."""
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()

    heat_map = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")
    heat_data = [
        [row.geometry.y, row.geometry.x, row.get("price_normalized", row["price"]) / 1_000_000]
        for _, row in gdf.iterrows()
    ]
    HeatMap(heat_data, radius=12, blur=18, max_zoom=12).add_to(heat_map)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    heat_map.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create Folium EDA maps for King County housing data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to cleaned GeoJSON dataset")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for HTML map outputs")
    args = parser.parse_args()

    gdf = load_processed_data(args.input)
    scatter_path = os.path.join(args.output_dir, "king_county_price_scatter.html")
    heatmap_path = os.path.join(args.output_dir, "king_county_price_heatmap.html")

    create_price_scatter_map(gdf, scatter_path)
    create_price_heatmap(gdf, heatmap_path)

    print(f"Saved scatter map: {scatter_path}")
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Properties mapped: {len(gdf):,}")


if __name__ == "__main__":
    main()
