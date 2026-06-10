"""Scaling helpers for neural forecasting models."""
import numpy as np
import pandas as pd

from common.models import SeriesScaler


def fit_series_scaler(series: pd.Series) -> SeriesScaler:
    """Fit a mean/std scaler on one spread series."""
    values = series.dropna().astype(float).to_numpy()
    if len(values) == 0:
        raise ValueError("Cannot fit scaler on an empty series.")
    mean = float(values.mean())
    scale = float(values.std(ddof=1))
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    return SeriesScaler(mean=mean, scale=scale)


def transform_series(series: pd.Series, scaler: SeriesScaler) -> pd.Series:
    """Apply a fitted scaler to a spread series."""
    values = (series.astype(float) - scaler.mean) / scaler.scale
    return pd.Series(values.to_numpy(dtype=float), index=series.index, name=series.name)


def inverse_transform_array(values: np.ndarray, scaler: SeriesScaler) -> np.ndarray:
    """Map scaled predictions back to spread units."""
    return values.astype(float) * scaler.scale + scaler.mean
