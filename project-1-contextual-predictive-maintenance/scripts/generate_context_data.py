"""Generate simulated external contextual data for predictive maintenance.

This script creates a timestamped dataset of environmental and operational signals
that can be merged with internal IoT telemetry data by timestamp.

Columns generated:
- timestamp
- ambient_temperature
- humidity
- factory_load

The dataset is saved to `data/external/context_data.csv` and summary output is printed
for quick validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_contextual_data(num_records: int, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic contextual dataset with environmental features.

    Args:
        num_records: Number of timestamped records to generate.
        seed: Random seed for reproducible results.

    Returns:
        A pandas DataFrame with timestamp, ambient_temperature, humidity, and factory_load.
    """
    rng = np.random.default_rng(seed)

    # Create a timestamp index with regular time spacing. This can later be aligned
    # to IoT telemetry timestamps by merge or join operations.
    timestamps = pd.date_range(start="2024-01-01 00:00:00", periods=num_records, freq="15T")

    ambient_temperature = rng.normal(loc=30.0, scale=5.0, size=num_records)
    ambient_temperature = np.clip(ambient_temperature, 20.0, 45.0)

    humidity = rng.normal(loc=60.0, scale=15.0, size=num_records)
    humidity = np.clip(humidity, 30.0, 90.0)

    # Factory load is modeled as a percentage that varies over time but stays within realistic bounds.
    load_base = rng.uniform(0.7, 0.95, size=num_records)
    factory_load = load_base * 100.0
    factory_load = np.clip(factory_load + rng.normal(loc=0.0, scale=5.0, size=num_records), 40.0, 100.0)

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ambient_temperature": ambient_temperature.round(2),
            "humidity": humidity.round(2),
            "factory_load": factory_load.round(2),
        }
    )

    return data


def save_contextual_data(data: pd.DataFrame, output_path: Path) -> None:
    """Save the contextual dataset to a CSV file.

    Args:
        data: DataFrame containing the simulated contextual data.
        output_path: Destination path for the CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def main() -> None:
    """Generate the contextual dataset and print a summary of the results."""
    num_records = 10_000
    output_path = Path(__file__).resolve().parents[1] / "data" / "external" / "context_data.csv"

    contextual_data = generate_contextual_data(num_records=num_records, seed=42)
    save_contextual_data(contextual_data, output_path)

    print("Contextual data generation complete.")
    print(f"Saved dataset to: {output_path}")
    print(f"Dataset shape: {contextual_data.shape}\n")
    print("First 5 rows:")
    print(contextual_data.head().to_string(index=False))
    print("\nSummary statistics:")
    print(contextual_data.describe(include="all"))


if __name__ == "__main__":
    main()
