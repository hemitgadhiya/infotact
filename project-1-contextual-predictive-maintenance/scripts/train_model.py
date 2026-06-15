"""Placeholder training script for predictive maintenance model."""

from pathlib import Path


def main() -> None:
    model_dir = Path(__file__).resolve().parents[1] / "models"
    print(f"Training placeholder; model_dir={model_dir}")


if __name__ == "__main__":
    main()
