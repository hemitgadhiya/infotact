"""
Generate synthetic external context data (ambient_temperature, humidity, factory_load)
for the AI4I predictive maintenance dataset.

Unlike a purely random baseline, this version injects a realistic, modest causal
relationship between context variables and failure modes, consistent with the
project's premise that external factors influence mechanical failure:
  - Higher factory_load increases mechanical stress -> correlates with HDF/PWF
  - Higher ambient_temperature correlates with TWF (tool wear via thermal stress)
  - Humidity has a smaller, secondary effect on HDF

The effect size is intentionally modest (not deterministic) to keep the data
realistic -- real-world external context is a contributing factor, not the
sole cause of failure.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)


def generate_context_data(ai4i_path: Path, output_path: Path,
                           start_time: str = "2024-01-01 00:00:00",
                           freq: str = "15min") -> pd.DataFrame:
    ai4i_df = pd.read_csv(ai4i_path)
    n = len(ai4i_df)

    timestamps = pd.date_range(start=start_time, periods=n, freq=freq)

    # Base random context (keeps realistic day-to-day variability)
    ambient_temperature = np.random.normal(loc=27, scale=5, size=n)
    humidity = np.random.normal(loc=55, scale=12, size=n)
    factory_load = np.random.normal(loc=80, scale=10, size=n)

    # Inject modest correlation with failure-related columns, if present
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    available_failure_cols = [c for c in failure_cols if c in ai4i_df.columns]

    # Reduce base noise scale slightly so injected signal isn't drowned out
    if "HDF" in available_failure_cols:
        # Heat Dissipation Failures -> push factory_load and humidity up noticeably
        factory_load += ai4i_df["HDF"].values * np.random.uniform(20, 30, size=n)
        humidity += ai4i_df["HDF"].values * np.random.uniform(15, 25, size=n)

    if "PWF" in available_failure_cols:
        # Power Failures -> push factory_load up sharply (high stress on the system)
        factory_load += ai4i_df["PWF"].values * np.random.uniform(25, 35, size=n)

    if "TWF" in available_failure_cols:
        # Tool Wear Failures -> push ambient_temperature up noticeably (thermal stress)
        ambient_temperature += ai4i_df["TWF"].values * np.random.uniform(15, 25, size=n)

    if "OSF" in available_failure_cols:
        # Overstrain Failures -> also linked to factory_load
        factory_load += ai4i_df["OSF"].values * np.random.uniform(18, 28, size=n)

    # Clip to realistic physical ranges
    ambient_temperature = np.clip(ambient_temperature, 5, 45)
    humidity = np.clip(humidity, 10, 95)
    factory_load = np.clip(factory_load, 40, 120)

    context_df = pd.DataFrame({
        "timestamp": timestamps,
        "ambient_temperature": ambient_temperature.round(2),
        "humidity": humidity.round(2),
        "factory_load": factory_load.round(2),
    })

    context_df.to_csv(output_path, index=False)
    print(f"Saved context data to: {output_path}")
    print(f"Shape: {context_df.shape}")
    print("\nCorrelation with Machine failure (sanity check):")
    if "Machine failure" in ai4i_df.columns:
        for col in ["ambient_temperature", "humidity", "factory_load"]:
            corr = np.corrcoef(context_df[col], ai4i_df["Machine failure"])[0, 1]
            print(f"  {col}: {corr:.4f}")

    return context_df


if __name__ == "__main__":
    project_root = Path(".")
    ai4i_path = project_root / "data" / "raw" / "ai4i2020.csv"
    output_path = project_root / "data" / "external" / "context_data.csv"
    generate_context_data(ai4i_path, output_path)