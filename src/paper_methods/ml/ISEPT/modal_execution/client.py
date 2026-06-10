"""Local client wrappers for Modal ISEPT training."""
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import modal
import pandas as pd

from common.config import ModalModelConfig
from modal_execution.app import app, check_tensorflow_gpu, train_and_select_isept_pairs


@contextmanager
def modal_app_context() -> Iterator[None]:
    """Run the ephemeral Modal app."""
    with modal.enable_output():
        with app.run():
            yield


def modal_gpu_status() -> dict[str, Any]:
    """Return TensorFlow GPU status from Modal."""
    return check_tensorflow_gpu.remote()


def train_select_pairs_on_modal(
    images,
    image_metadata: list[dict],
    label_frame: pd.DataFrame,
    config: ModalModelConfig,
    seed: int,
) -> dict[str, Any]:
    """Train ISEPT CAE/MLP on Modal and return selected pairs."""
    result = train_and_select_isept_pairs.remote(
        images,
        image_metadata,
        label_frame.to_dict("records"),
        modal_config_to_payload(config),
        seed,
    )
    return {
        "selected_pairs": pd.DataFrame(result["selected_pairs"]),
        "cae_history": result["cae_history"],
        "embedding_count": result["embedding_count"],
    }


def modal_config_to_payload(config: ModalModelConfig) -> dict[str, Any]:
    """Convert model config into a Modal payload."""
    return vars(config).copy()
