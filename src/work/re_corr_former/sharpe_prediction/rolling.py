"""ISEPT-style rolling feedback training for Sharpe prediction."""
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from common.config import ScoringConfig, SharpeModelConfig
from sharpe_prediction.datasets import PAIR_SCALAR_NAMES, SharpeDataset, normalize_inputs, subset_inputs
from sharpe_prediction.models import build_sharpe_model


def rolling_sharpe_predictions(
    dataset: SharpeDataset,
    model_config: SharpeModelConfig,
    scoring_config: ScoringConfig,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Train rolling Sharpe models and predict candidate Sharpe by month."""
    selected_prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict] = []
    months = sorted(dataset.metadata["label_month"].unique())

    for month_index, month in enumerate(months):
        if month_index < model_config.warmup_months:
            continue
        training_rows = feedback_training_rows(dataset.metadata[dataset.metadata["label_month"] < month], model_config)
        current_rows = dataset.metadata[dataset.metadata["label_month"] == month]
        if training_rows.empty or current_rows.empty:
            continue
        train_indices = training_rows["sample_index"].to_numpy(dtype=int)
        current_indices = current_rows["sample_index"].to_numpy(dtype=int)
        train_inputs = subset_inputs(dataset.inputs, train_indices)
        current_inputs = subset_inputs(dataset.inputs, current_indices)
        train_inputs, current_inputs, _ = normalize_inputs(train_inputs, current_inputs)
        x_fit, x_validation, y_fit, y_validation = split_supervised_inputs(
            train_inputs,
            dataset.targets[train_indices],
            seed + month_index,
        )
        model = build_sharpe_model(sequence_shape(dataset), len(PAIR_SCALAR_NAMES), model_config, seed + month_index)
        validation_data = (x_validation, y_validation) if len(y_validation) else None
        history = model.fit(
            x_fit,
            y_fit,
            validation_data=validation_data,
            epochs=model_config.epochs,
            batch_size=model_config.batch_size,
            callbacks=training_callbacks(model_config) if validation_data else None,
            verbose=0,
            shuffle=True,
        )
        history_rows.extend(history_records(month, history))
        current_prediction = current_rows.reset_index(drop=True).copy()
        current_prediction["predicted_sharpe"] = model.predict(current_inputs, verbose=0).reshape(-1)
        current_prediction["final_score"] = current_prediction["predicted_sharpe"]
        current_prediction["split"] = "rolling_test"
        selected_prediction_frames.append(current_prediction)

    save_history(history_rows, output_dir / "model" / f"sharpe_{model_config.model_type}_history.csv")
    if not selected_prediction_frames:
        return pd.DataFrame()
    return pd.concat(selected_prediction_frames, axis=0, ignore_index=True)


def feedback_training_rows(metadata: pd.DataFrame, config: SharpeModelConfig) -> pd.DataFrame:
    """Keep top and bottom realized-Sharpe rows per feedback month."""
    rows = []
    per_side = config.feedback_pairs_per_side
    for _, group in metadata.groupby("label_month", sort=True):
        rows.append(group.nlargest(per_side, "realized_sharpe"))
        rows.append(group.nsmallest(per_side, "realized_sharpe"))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True).drop_duplicates(["label_month", "asset_i", "asset_j"])


def split_supervised_inputs(inputs: dict[str, np.ndarray], targets: np.ndarray, seed: int):
    """Split supervised inputs into 70/30 fit and validation sets like ISEPT."""
    count = len(targets)
    if count < 2:
        empty_inputs = {name: values[:0] for name, values in inputs.items()}
        return inputs, empty_inputs, targets, targets[:0]
    order = np.random.default_rng(seed).permutation(count)
    split = max(1, int(count * 0.70))
    if split >= count:
        split = count - 1
    fit_indices = order[:split]
    validation_indices = order[split:]
    x_fit = {name: values[fit_indices] for name, values in inputs.items()}
    x_validation = {name: values[validation_indices] for name, values in inputs.items()}
    return x_fit, x_validation, targets[fit_indices], targets[validation_indices]


def training_callbacks(config: SharpeModelConfig) -> list:
    """Return ISEPT-style early stopping callbacks."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        )
    ]


def history_records(month: str, history) -> list[dict]:
    """Return flat history rows for one rolling month."""
    rows = []
    epoch_count = len(next(iter(history.history.values())))
    for epoch in range(epoch_count):
        row = {"label_month": month, "epoch": epoch + 1}
        for key, values in history.history.items():
            row[key] = float(values[epoch])
        rows.append(row)
    return rows


def save_history(rows: list[dict], output_path: Path) -> None:
    """Persist rolling training history."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def sequence_shape(dataset: SharpeDataset) -> tuple[int, int]:
    """Return the OHLCV sequence shape."""
    values = dataset.inputs["asset_i_ohlcv"]
    return int(values.shape[1]), int(values.shape[2])


def select_sharpe_pairs_by_month(predictions: pd.DataFrame, scoring_config: ScoringConfig) -> pd.DataFrame:
    """Select top predicted-Sharpe pairs per month with optional no-overlap matching."""
    selected_rows: list[dict] = []
    for _, group in predictions.groupby("label_month", sort=True):
        used_assets: set[str] = set()
        ranked = group.sort_values("predicted_sharpe", ascending=False)
        rank = 0
        for _, row in ranked.iterrows():
            asset_i = str(row["asset_i"])
            asset_j = str(row["asset_j"])
            if not scoring_config.allow_asset_reuse and (asset_i in used_assets or asset_j in used_assets):
                continue
            rank += 1
            selected = row.to_dict()
            selected["selection_rank"] = rank
            selected["final_score"] = float(row["predicted_sharpe"])
            selected_rows.append(selected)
            used_assets.update([asset_i, asset_j])
            if rank >= scoring_config.top_k:
                break
    return pd.DataFrame(selected_rows)
