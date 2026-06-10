"""Build OHLCV plus correlation datasets for Sharpe prediction."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.config import SharpeModelConfig, TradeConfig
from evaluation.metrics import sharpe_ratio
from trading.rule_backtest import run_trade_event


OHLCV_FEATURE_NAMES = ("open", "high", "low", "close", "volume")
PAIR_SCALAR_NAMES = ("rho_long", "rho_now", "gap")


@dataclass
class SharpeDataset:
    inputs: dict[str, np.ndarray]
    targets: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class ScalarNormalizer:
    mean: np.ndarray
    scale: np.ndarray


def build_sharpe_dataset(
    candidate_metadata: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    trade_config: TradeConfig,
    model_config: SharpeModelConfig,
) -> SharpeDataset:
    """Convert candidate rows into Sharpe model inputs and realized-Sharpe labels."""
    asset_i_rows: list[np.ndarray] = []
    asset_j_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    metadata_rows: list[dict] = []

    for _, row in candidate_metadata.iterrows():
        date = pd.Timestamp(row["date"])
        if date not in close.index:
            continue
        position = int(close.index.get_loc(date))
        if position - model_config.lookback + 1 < 0:
            continue
        sample_index = len(target_rows)
        label = realized_sharpe_label(sample_index + 1, row, close, trade_config)
        asset_i_rows.append(ohlcv_tensor_for_asset(panel, str(row["asset_i"]), date, model_config.lookback))
        asset_j_rows.append(ohlcv_tensor_for_asset(panel, str(row["asset_j"]), date, model_config.lookback))
        scalar_rows.append(pair_scalar_values(row))
        target_rows.append(label["realized_sharpe"])
        metadata_rows.append(sharpe_metadata_row(sample_index, row, label))

    return SharpeDataset(
        inputs={
            "asset_i_ohlcv": np.asarray(asset_i_rows, dtype=np.float32),
            "asset_j_ohlcv": np.asarray(asset_j_rows, dtype=np.float32),
            "pair_scalars": np.asarray(scalar_rows, dtype=np.float32),
        },
        targets=np.asarray(target_rows, dtype=np.float32),
        metadata=pd.DataFrame(metadata_rows),
    )


def ohlcv_tensor_for_asset(
    panel: dict[str, pd.DataFrame],
    asset: str,
    end_date: pd.Timestamp,
    lookback: int,
) -> np.ndarray:
    """Return a normalized OHLCV tensor ending at end_date."""
    frame = panel[asset].loc[:, list(OHLCV_FEATURE_NAMES)]
    end_position = int(frame.index.get_loc(end_date))
    start_position = end_position - lookback + 1
    window = frame.iloc[start_position:end_position + 1].astype(float)
    prices = normalized_price_columns(window)
    volume = normalized_volume_column(window["volume"])
    tensor = np.column_stack([prices, volume])
    return np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalized_price_columns(window: pd.DataFrame) -> np.ndarray:
    """Normalize OHLC prices by the first close in the lookback window."""
    base = float(window["close"].iloc[0])
    if base == 0.0 or not np.isfinite(base):
        base = 1.0
    price_columns = window.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
    return price_columns / base - 1.0


def normalized_volume_column(volume: pd.Series) -> np.ndarray:
    """Return z-scored log volume for one lookback window."""
    values = np.log1p(volume.astype(float).to_numpy(dtype=float))
    mean = float(np.nanmean(values))
    scale = float(np.nanstd(values))
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    return ((values - mean) / scale).reshape(-1, 1)


def pair_scalar_values(row: pd.Series) -> np.ndarray:
    """Return pair-level short/long correlation inputs."""
    return np.asarray([float(row[name]) for name in PAIR_SCALAR_NAMES], dtype=np.float32)


def realized_sharpe_label(event_number: int, row: pd.Series, close: pd.DataFrame, trade_config: TradeConfig) -> dict:
    """Compute the realized future trading Sharpe label for one candidate pair."""
    label_row = row.copy()
    label_row["final_score"] = 0.0
    label_row["predicted_sharpe"] = 0.0
    result = run_trade_event(event_number, label_row, close, trade_config)
    returns = result["returns"]
    trades = result["trades"]
    total_return = float(returns.add(1.0).prod() - 1.0) if not returns.empty else 0.0
    return {
        "realized_sharpe": sharpe_ratio(returns),
        "realized_total_return_pct": total_return * 100.0,
        "realized_trade_count": int(len(trades)),
    }


def sharpe_metadata_row(sample_index: int, row: pd.Series, label: dict) -> dict:
    """Return metadata for one Sharpe training sample."""
    values = row.to_dict()
    date = pd.Timestamp(values["date"])
    values["sample_index"] = sample_index
    values["label_month"] = date.to_period("M").strftime("%Y-%m")
    values.update(label)
    return values


def fit_scalar_normalizer(values: np.ndarray) -> ScalarNormalizer:
    """Fit scalar normalization statistics."""
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return ScalarNormalizer(mean=mean.astype(np.float32), scale=scale.astype(np.float32))


def apply_scalar_normalizer(values: np.ndarray, normalizer: ScalarNormalizer) -> np.ndarray:
    """Apply scalar normalization statistics."""
    return ((values - normalizer.mean) / normalizer.scale).astype(np.float32)


def subset_inputs(inputs: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    """Return indexed model inputs."""
    return {name: values[indices] for name, values in inputs.items()}


def normalize_inputs(
    train_inputs: dict[str, np.ndarray],
    candidate_inputs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], ScalarNormalizer]:
    """Normalize pair scalar inputs with training samples only."""
    normalizer = fit_scalar_normalizer(train_inputs["pair_scalars"])
    normalized_train = dict(train_inputs)
    normalized_candidate = dict(candidate_inputs)
    normalized_train["pair_scalars"] = apply_scalar_normalizer(train_inputs["pair_scalars"], normalizer)
    normalized_candidate["pair_scalars"] = apply_scalar_normalizer(candidate_inputs["pair_scalars"], normalizer)
    return normalized_train, normalized_candidate, normalizer
