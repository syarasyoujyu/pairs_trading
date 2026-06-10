"""Train and predict with ReCorrFormer."""
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from common.config import ModelConfig
from models.re_corr_former import build_re_corr_former, set_tensorflow_seed
from training.datasets import SCALAR_FEATURE_NAMES, SupervisedDataset, sequence_shape


def train_re_corr_former(
    splits: dict[str, SupervisedDataset],
    lookback: int,
    config: ModelConfig,
    seed: int,
    output_dir: Path,
) -> keras.Model:
    """Train ReCorrFormer and persist the model and history."""
    set_tensorflow_seed(seed)
    model = build_re_corr_former(sequence_shape(lookback), len(SCALAR_FEATURE_NAMES), config)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    history = model.fit(
        splits["train"].inputs,
        splits["train"].targets,
        validation_data=(splits["validation"].inputs, splits["validation"].targets),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "re_corr_former.keras")
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)
    return model


def predict_re_corr_former(model: keras.Model, dataset: SupervisedDataset) -> pd.DataFrame:
    """Predict future correlation recovery for a dataset."""
    raw = model.predict(dataset.inputs, verbose=0)
    frame = dataset.metadata.copy()
    frame["pred_corr"] = np.asarray(raw["corr"], dtype=float).reshape(-1)
    return frame
