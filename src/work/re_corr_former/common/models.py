"""Data containers used across ReCorrFormer steps."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CandidatePair:
    date: pd.Timestamp
    asset_i: str
    asset_j: str
    rho_long: float
    rho_now: float
    gap: float
    long_corr_rank: int
    long_corr_rank_fraction: float
    long_corr_pair_count: int
    long_corr_start_date: pd.Timestamp
    long_corr_end_date: pd.Timestamp
    short_corr_start_date: pd.Timestamp
    short_corr_end_date: pd.Timestamp
    spread_z: float
    spread_volatility: float
    beta: float


@dataclass(frozen=True)
class PairLabels:
    y_corr: float
    best_horizon: int


@dataclass(frozen=True)
class DatasetSplits:
    train_end_date: pd.Timestamp
    validation_end_date: pd.Timestamp


@dataclass(frozen=True)
class ForecastThresholds:
    name: str
    short_entry: float
    long_entry: float


@dataclass(frozen=True)
class SeriesScaler:
    mean: float
    scale: float


@dataclass
class SpreadForecastModel:
    name: str
    model: object
    scaler: SeriesScaler
    input_length: int
    horizon: int
    history: object | None = None
