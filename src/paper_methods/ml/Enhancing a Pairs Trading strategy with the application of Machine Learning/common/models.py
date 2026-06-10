"""Data containers for the Sarmento and Horta (2020) method."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CointegrationTestResult:
    dependent: str
    independent: str
    hedge_ratio: float
    intercept: float
    t_stat: float
    p_value: float | None
    critical_value_5pct: float | None
    is_cointegrated: bool
    spread: pd.Series


@dataclass(frozen=True)
class PairDiagnostics:
    asset_y: str
    asset_x: str
    hedge_ratio: float
    intercept: float
    adf_t_stat: float
    adf_p_value: float | None
    hurst: float
    half_life_bars: float
    crossings_per_year: float
    spread_mean: float
    spread_std: float
    is_selected: bool
    rejection_reason: str


@dataclass(frozen=True)
class PairCandidate:
    asset_y: str
    asset_x: str
    hedge_ratio: float
    intercept: float
    diagnostics: PairDiagnostics
    spread: pd.Series


@dataclass(frozen=True)
class SelectedPair:
    asset_y: str
    asset_x: str
    hedge_ratio: float
    intercept: float
    diagnostics: PairDiagnostics
    spread: pd.Series


@dataclass(frozen=True)
class StandardThresholds:
    long_entry: float
    short_entry: float
    exit: float


@dataclass(frozen=True)
class ForecastThresholds:
    name: str
    short_entry: float
    long_entry: float


@dataclass(frozen=True)
class SeriesScaler:
    mean: float
    scale: float


@dataclass(frozen=True)
class ARMAForecastConfig:
    ar_order: int = 8
    ma_order: int = 3
    horizon: int = 1


@dataclass(frozen=True)
class LSTMForecastConfig:
    input_length: int = 24
    horizon: int = 1
    hidden_units: int = 50
    hidden_layers: int = 1
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 64
    patience: int = 5
    learning_rate: float = 0.001


@dataclass(frozen=True)
class EncoderDecoderForecastConfig:
    input_length: int = 24
    horizon: int = 2
    encoder_units: int = 30
    decoder_units: int = 30
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 64
    patience: int = 5
    learning_rate: float = 0.001


@dataclass
class NeuralForecastModel:
    name: str
    model: object
    scaler: SeriesScaler
    input_length: int
    horizon: int
    history: object | None = None
