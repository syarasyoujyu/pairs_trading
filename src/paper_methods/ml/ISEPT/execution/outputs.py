"""Output helpers for ISEPT."""
from pathlib import Path

import pandas as pd


def save_frame(frame: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def save_series(series: pd.Series, output_path: Path) -> None:
    """Save a Series to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(output_path, index_label="date")


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save one metrics row."""
    save_frame(pd.DataFrame([metrics]), output_path)
