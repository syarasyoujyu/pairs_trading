"""Local client wrappers for Modal ReCorrFormer execution."""
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import modal

from common.config import ModelConfig
from modal_execution.app import app, check_tensorflow_gpu, train_and_predict_re_corr_former
from modal_execution.payloads import frame_from_records, model_config_to_payload, split_datasets_to_payload
from training.datasets import SupervisedDataset


@contextmanager
def modal_app_context() -> Iterator[None]:
    """Run the ephemeral Modal app while local code calls remote functions."""
    with modal.enable_output():
        with app.run():
            yield


def modal_gpu_status() -> dict[str, Any]:
    """Return TensorFlow GPU status from Modal."""
    return check_tensorflow_gpu.remote()


def train_predict_re_corr_former_on_modal(
    splits: dict[str, SupervisedDataset],
    lookback: int,
    config: ModelConfig,
    seed: int,
) -> dict[str, Any]:
    """Train and predict on Modal, returning local pandas outputs."""
    result = train_and_predict_re_corr_former.remote(
        split_datasets_to_payload(splits),
        lookback,
        model_config_to_payload(config),
        seed,
    )
    return {
        "history": result["history"],
        "predictions": frame_from_records(result["predictions"]),
    }
