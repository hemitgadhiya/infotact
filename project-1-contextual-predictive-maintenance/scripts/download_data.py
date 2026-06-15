"""Placeholder script for dataset download in project 1."""

from __future__ import annotations

import os
from pathlib import Path


def download_data(data_dir: Path) -> None:
    """Create the target data directory and print a placeholder message."""
    data_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Placeholder: dataset download logic will be implemented here. "
        f"Data directory is {data_dir.resolve()}"
    )


def main() -> None:
    """Entry point for the download_data script."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    download_data(data_dir)


if __name__ == "__main__":
    main()
