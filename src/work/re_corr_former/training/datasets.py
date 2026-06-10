"""Build supervised arrays for ReCorrFormer."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.config import ModelConfig, WindowConfig
from common.models import CandidatePair
from features.asset_features import FEATURE_NAMES, feature_tensor_for_asset
from labels.recovery_labels import label_candidate


SCALAR_FEATURE_NAMES = ("rho_long", "rho_now", "gap", "spread_z", "spread_volatility", "beta")


@dataclass
class SupervisedDataset:
    inputs: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    metadata: pd.DataFrame


@dataclass(frozen=True)
class ScalarNormalizer:
    mean: np.ndarray
    scale: np.ndarray


def build_supervised_dataset(
    candidates: list[CandidatePair],
    features: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    returns: pd.DataFrame,
    windows: WindowConfig,
) -> SupervisedDataset:
    """Convert candidates into model-ready arrays and labels."""
    asset_i_rows: list[np.ndarray] = []
    asset_j_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    target_rows: dict[str, list] = {"corr": []}
    metadata_rows: list[dict] = []

    for candidate in candidates:
        feature_position = int(features["return"].index.get_loc(candidate.date))
        labels = label_candidate(candidate, close, returns, windows)
        asset_i_rows.append(feature_tensor_for_asset(features, candidate.asset_i, feature_position, windows.lookback))
        asset_j_rows.append(feature_tensor_for_asset(features, candidate.asset_j, feature_position, windows.lookback))
        scalar_rows.append(candidate_scalar_features(candidate))
        target_rows["corr"].append(labels.y_corr)
        metadata_rows.append(candidate_metadata(candidate, labels.y_corr, labels.best_horizon))

    inputs = {
        "asset_i_sequence": np.asarray(asset_i_rows, dtype=np.float32),
        "asset_j_sequence": np.asarray(asset_j_rows, dtype=np.float32),
        "pair_scalars": np.asarray(scalar_rows, dtype=np.float32),
    }
    targets = {
        "corr": np.asarray(target_rows["corr"], dtype=np.float32),
    }
    return SupervisedDataset(inputs=inputs, targets=targets, metadata=pd.DataFrame(metadata_rows))


def split_supervised_dataset(dataset: SupervisedDataset, model_config: ModelConfig) -> dict[str, SupervisedDataset]:
    """Chronologically split samples into train, validation, and test sets."""
    dates = pd.to_datetime(dataset.metadata["date"])
    unique_dates = pd.Index(sorted(dates.unique()))
    train_count = max(1, int(len(unique_dates) * (1.0 - 2.0 * model_config.validation_fraction)))
    validation_count = max(1, int(len(unique_dates) * model_config.validation_fraction))
    train_dates = set(unique_dates[:train_count])
    validation_dates = set(unique_dates[train_count:train_count + validation_count])

    masks = {
        "train": dates.isin(train_dates).to_numpy(),
        "validation": dates.isin(validation_dates).to_numpy(),
        "test": (~dates.isin(train_dates) & ~dates.isin(validation_dates)).to_numpy(),
    }
    return {name: subset_supervised_dataset(dataset, mask) for name, mask in masks.items()}


def add_high_corr_label_columns(dataset: SupervisedDataset, top_fraction: float) -> SupervisedDataset:
    """Add high-future-correlation labels to metadata."""
    metadata = dataset.metadata.copy()
    metadata["y_corr_rank_on_date"] = 0
    metadata["y_corr_rank_fraction"] = 0.0
    metadata["is_high_y_corr"] = False
    for _, group in metadata.groupby("date", sort=False):
        ranked_index = group.sort_values("y_corr", ascending=False).index
        total = max(1, len(ranked_index))
        top_count = max(1, int(np.ceil(total * min(max(top_fraction, 0.0), 1.0))))
        for rank, row_index in enumerate(ranked_index, start=1):
            metadata.loc[row_index, "y_corr_rank_on_date"] = rank
            metadata.loc[row_index, "y_corr_rank_fraction"] = rank / total
            metadata.loc[row_index, "is_high_y_corr"] = rank <= top_count and metadata.loc[row_index, "y_corr"] > 0.0
    return SupervisedDataset(inputs=dataset.inputs, targets=dataset.targets, metadata=metadata)


def normalize_split_scalars(splits: dict[str, SupervisedDataset]) -> tuple[dict[str, SupervisedDataset], ScalarNormalizer]:
    """Normalize pair scalar inputs using training samples only."""
    normalizer = fit_scalar_normalizer(splits["train"].inputs["pair_scalars"])
    for split in splits.values():
        split.inputs["pair_scalars"] = apply_scalar_normalizer(split.inputs["pair_scalars"], normalizer)
    return splits, normalizer


def fit_scalar_normalizer(values: np.ndarray) -> ScalarNormalizer:
    """Fit scalar mean and scale."""
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return ScalarNormalizer(mean=mean.astype(np.float32), scale=scale.astype(np.float32))


def apply_scalar_normalizer(values: np.ndarray, normalizer: ScalarNormalizer) -> np.ndarray:
    """Apply scalar normalization."""
    return ((values - normalizer.mean) / normalizer.scale).astype(np.float32)


def subset_supervised_dataset(dataset: SupervisedDataset, mask: np.ndarray) -> SupervisedDataset:
    """Take a boolean-mask subset of a supervised dataset."""
    inputs = {name: values[mask] for name, values in dataset.inputs.items()}
    targets = {name: values[mask] for name, values in dataset.targets.items()}
    metadata = dataset.metadata.loc[mask].reset_index(drop=True)
    return SupervisedDataset(inputs=inputs, targets=targets, metadata=metadata)


def candidate_scalar_features(candidate: CandidatePair) -> np.ndarray:
    """Return scalar pair features used by the pair scoring network."""
    return np.asarray(
        [
            candidate.rho_long,
            candidate.rho_now,
            candidate.gap,
            candidate.spread_z,
            candidate.spread_volatility,
            candidate.beta,
        ],
        dtype=np.float32,
    )


def candidate_metadata(
    candidate: CandidatePair,
    y_corr: float,
    best_horizon: int,
) -> dict:
    """Return one metadata row for outputs and evaluation."""
    return {
        "date": candidate.date,
        "asset_i": candidate.asset_i,
        "asset_j": candidate.asset_j,
        "rho_long": candidate.rho_long,
        "rho_now": candidate.rho_now,
        "gap": candidate.gap,
        "long_corr_rank": candidate.long_corr_rank,
        "long_corr_rank_fraction": candidate.long_corr_rank_fraction,
        "long_corr_pair_count": candidate.long_corr_pair_count,
        "long_corr_start_date": candidate.long_corr_start_date,
        "long_corr_end_date": candidate.long_corr_end_date,
        "short_corr_start_date": candidate.short_corr_start_date,
        "short_corr_end_date": candidate.short_corr_end_date,
        "spread_z": candidate.spread_z,
        "spread_volatility": candidate.spread_volatility,
        "beta": candidate.beta,
        "y_corr": y_corr,
        "best_horizon": best_horizon,
    }


def sequence_shape(lookback: int) -> tuple[int, int]:
    """Return the expected asset sequence shape."""
    return lookback, len(FEATURE_NAMES)
