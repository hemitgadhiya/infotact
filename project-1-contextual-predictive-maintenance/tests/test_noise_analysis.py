import pandas as pd

from src.noise_analysis import evaluate_noise_sensitivity, inject_sensor_noise


def test_inject_sensor_noise_changes_only_selected_columns():
    df = pd.DataFrame(
        {
            "sensor_a": [1.0, 2.0, 3.0],
            "sensor_b": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        }
    )

    noisy = inject_sensor_noise(df, noise_level=0.1, sensor_columns=["sensor_a"], random_state=42)

    assert noisy.shape == df.shape
    assert noisy["sensor_a"].notna().all()
    assert noisy["sensor_b"].equals(df["sensor_b"])
    assert (noisy["sensor_a"] != df["sensor_a"]).any()


def test_evaluate_noise_sensitivity_returns_noise_levels_and_metrics():
    df = pd.DataFrame(
        {
            "sensor_a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "sensor_b": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )

    results = evaluate_noise_sensitivity(
        df,
        target_column="target",
        sensor_columns=["sensor_a", "sensor_b"],
        noise_levels=[0.0, 0.1],
        random_state=42,
    )

    assert list(results["noise_level"]) == [0.0, 0.1]
    assert {"macro_f1"}.issubset(results.columns)
    assert results.iloc[0]["macro_f1"] >= results.iloc[1]["macro_f1"]
