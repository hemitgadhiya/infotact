"""Utility functions for the predictive maintenance project."""


def ensure_directory(path):
    """Create directory if it does not exist."""
    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)
