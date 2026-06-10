"""Modal app for GPU-backed ReCorrFormer training and inference."""
from pathlib import Path
import sys
from typing import Any

import modal


APP_NAME = "recorr-former-research-plan"
LOCAL_METHOD_DIR = Path(__file__).resolve().parents[1]
REMOTE_METHOD_DIR = Path("/root/re_corr_former")
GPU_FALLBACKS = ["L4", "T4"]
REMOTE_TIMEOUT_SECONDS = 60 * 60
PYTHON_SITE_PACKAGES = "/usr/local/lib/python3.13/site-packages"
CUDA_LIBRARY_PATHS = [
    f"{PYTHON_SITE_PACKAGES}/nvidia/cublas/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/cuda_cupti/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/cufft/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/curand/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/cusolver/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/cusparse/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/nccl/lib",
    f"{PYTHON_SITE_PACKAGES}/nvidia/nvjitlink/lib",
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("pandas>=3.0.3", "tensorflow[and-cuda]>=2.21.0")
    .env({"LD_LIBRARY_PATH": ":".join(CUDA_LIBRARY_PATHS)})
    .add_local_dir(
        LOCAL_METHOD_DIR,
        remote_path=str(REMOTE_METHOD_DIR),
        ignore=["**/__pycache__/**", "*.pyc"],
    )
)
app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU_FALLBACKS, timeout=REMOTE_TIMEOUT_SECONDS)
def check_tensorflow_gpu() -> dict[str, Any]:
    """Return TensorFlow GPU visibility inside Modal."""
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    return {
        "tensorflow_version": tf.__version__,
        "gpu_devices": [device.name for device in devices],
        "gpu_count": len(devices),
    }


@app.function(gpu=GPU_FALLBACKS, timeout=REMOTE_TIMEOUT_SECONDS)
def train_and_predict_re_corr_former(
    splits_payload: dict[str, dict],
    lookback: int,
    config_payload: dict,
    seed: int,
) -> dict[str, Any]:
    """Train ReCorrFormer on Modal and return predictions for every split."""
    _prepare_remote_imports()

    import pandas as pd
    from tensorflow import keras

    from modal_execution.payloads import (
        frame_to_records,
        model_config_from_payload,
        split_datasets_from_payload,
    )
    from models.re_corr_former import build_re_corr_former, set_tensorflow_seed
    from training.datasets import SCALAR_FEATURE_NAMES, sequence_shape
    from training.train_model import predict_re_corr_former

    config = model_config_from_payload(config_payload)
    splits = split_datasets_from_payload(splits_payload)
    set_tensorflow_seed(seed)
    model = build_re_corr_former(sequence_shape(lookback), len(SCALAR_FEATURE_NAMES), config)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    history = model.fit(
        splits["train"].inputs,
        splits["train"].targets,
        validation_data=(splits["validation"].inputs, splits["validation"].targets),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    frames = []
    for split_name, split in splits.items():
        predictions = predict_re_corr_former(model, split)
        predictions["split"] = split_name
        frames.append(predictions)

    return {
        "history": _history_to_payload(history),
        "predictions": frame_to_records(pd.concat(frames, axis=0, ignore_index=True)),
    }


def _prepare_remote_imports() -> None:
    remote_path = str(REMOTE_METHOD_DIR)
    if remote_path not in sys.path:
        sys.path.insert(0, remote_path)


def _history_to_payload(history) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}
