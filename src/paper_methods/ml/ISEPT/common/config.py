"""Configuration for the ISEPT implementation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    interval: str = "1d"
    output_name: str = "ISEPT"
    max_assets: int = 20
    min_history_years: float = 8.0
    months: int = 8
    random_seed: int = 7


@dataclass(frozen=True)
class ImageConfig:
    image_size: int = 64
    candle_window: int = 21
    window_step: int = 1
    lookback_bars: int = 252


@dataclass(frozen=True)
class ModalModelConfig:
    cae_latent_dim: int = 16_384
    cae_base_filters: int = 64
    cae_epochs: int = 20
    cae_learning_rate: float = 0.0001
    cae_lr_decay_epochs: int = 5
    cae_lr_decay_factor: float = 0.5
    cae_patience: int = 3
    pca_components: int = 512
    mlp_epochs: int = 20
    mlp_learning_rate: float = 0.001
    mlp_patience: int = 5
    batch_size: int = 512
    feedback_pairs_per_side: int = 20
    warmup_months: int = 2
    top_k_pairs: int = 100
    allow_asset_reuse: bool = True


@dataclass(frozen=True)
class TradingConfig:
    trade_rule: str = "vidyamurthy"
    trading_horizon_months: int = 6
    entry_sigma: float = 2.0
    exit_sigma: float = 1.0
    vidyamurthy_threshold_sigma: float = 0.75
    transaction_cost_bps: float = 1.0
