"""Construct per-asset sequence features for ReCorrFormer."""
import numpy as np
import pandas as pd


FEATURE_NAMES = ("return", "volatility", "log_dollar_volume", "ma_gap", "momentum")


def build_asset_features(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create normalized feature matrices keyed by feature name."""
    returns = close.astype(float).pct_change()
    volatility = returns.rolling(20).std()
    dollar_volume = close.astype(float) * volume.astype(float)
    log_dollar_volume = np.log1p(dollar_volume)
    moving_average = close.astype(float).rolling(20).mean()
    ma_gap = close.astype(float).divide(moving_average).subtract(1.0)
    momentum = close.astype(float).pct_change(20)

    raw_features = {
        "return": returns,
        "volatility": volatility,
        "log_dollar_volume": log_dollar_volume,
        "ma_gap": ma_gap,
        "momentum": momentum,
    }
    return {name: standardize_feature(frame) for name, frame in raw_features.items()}


def standardize_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize a feature matrix using all available training-style observations."""
    values = frame.replace([np.inf, -np.inf], np.nan)
    mean = float(values.stack().mean())
    std = float(values.stack().std(ddof=1))
    scale = std if std > 0.0 else 1.0
    return values.subtract(mean).divide(scale)


def feature_tensor_for_asset(
    features: dict[str, pd.DataFrame],
    asset: str,
    end_position: int,
    lookback: int,
) -> np.ndarray:
    """Return a lookback x feature_count tensor ending at end_position."""
    start = end_position - lookback + 1
    if start < 0:
        raise ValueError("Not enough observations for the requested lookback.")
    columns = []
    for name in FEATURE_NAMES:
        columns.append(features[name].iloc[start:end_position + 1][asset].to_numpy(dtype=float))
    tensor = np.vstack(columns).T
    return np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def usable_feature_dates(features: dict[str, pd.DataFrame]) -> pd.Index:
    """Return dates where every feature has an index entry."""
    index = next(iter(features.values())).index
    for frame in features.values():
        index = index.intersection(frame.index)
    return index
