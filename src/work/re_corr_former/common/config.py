"""Configuration for the ReCorrFormer research-plan implementation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowConfig:
    lookback: int = 60
    long_corr_window: int = 120
    long_corr_min_lag: int = 15
    short_corr_window: int = 15
    short_corr_min_lag: int = 1
    future_min_horizon: int = 2
    future_max_horizon: int = 30
    future_corr_window: int = 10
    sample_stride: int = 10


@dataclass(frozen=True)
class CandidateConfig:
    min_long_corr: float = -1.0
    long_corr_top_fraction: float = 0.15
    min_gap: float = 0.0
    min_liquidity_quantile: float = 0.30
    max_candidates_per_date: int = 80


@dataclass(frozen=True)
class ModelConfig:
    encoder_units: int = 16
    dense_units: int = 32
    dense_layers: int = 1
    dropout: float = 0.20
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 2
    validation_fraction: float = 0.20


@dataclass(frozen=True)
class SharpeModelConfig:
    model_type: str = "lstm"
    lookback: int = 60
    encoder_units: int = 32
    dense_units: int = 64
    dense_layers: int = 2
    transformer_heads: int = 2
    transformer_ff_dim: int = 64
    transformer_layers: int = 1
    dropout: float = 0.20
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 5
    patience: int = 3
    warmup_months: int = 2
    feedback_pairs_per_side: int = 20


@dataclass(frozen=True)
class ScoringConfig:
    top_k: int = 20
    allow_asset_reuse: bool = False
    high_corr_top_fraction: float = 0.15


@dataclass(frozen=True)
class TradeConfig:
    trade_rule: str = "vidyamurthy"
    formation_window: int = 252
    trading_horizon_months: int = 6
    entry_sigma: float = 2.0
    exit_sigma: float = 1.0
    vidyamurthy_threshold_sigma: float = 0.75
    transaction_cost_bps: float = 1.0


@dataclass(frozen=True)
class RuntimeConfig:
    interval: str = "1d"
    output_name: str = "ReCorrFormer"
    max_assets: int = 40
    min_history_years: float = 8.0
    random_seed: int = 7
    backend: str = "local"
    selection_model: str = "corr"
