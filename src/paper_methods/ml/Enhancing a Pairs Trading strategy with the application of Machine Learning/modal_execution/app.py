"""Modal app for GPU-backed neural spread forecasting."""
import json
from pathlib import Path
import sys
from typing import Any

import modal


APP_NAME = "enhancing-pairs-trading-ml"
LOCAL_METHOD_DIR = Path(__file__).resolve().parents[1]
REMOTE_METHOD_DIR = Path("/root/enhancing_pairs_trading_ml")
MODEL_VOLUME_PATH = Path("/model_store")
MODEL_VOLUME_NAME = "enhancing-pairs-trading-ml-models"
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
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(gpu=GPU_FALLBACKS, timeout=REMOTE_TIMEOUT_SECONDS)
def check_tensorflow_gpu() -> dict[str, Any]:
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    return {
        "tensorflow_version": tf.__version__,
        "gpu_devices": [device.name for device in devices],
        "gpu_count": len(devices),
    }


@app.function(
    gpu=GPU_FALLBACKS,
    timeout=REMOTE_TIMEOUT_SECONDS,
    volumes={str(MODEL_VOLUME_PATH): model_volume},
)
def train_and_predict_spread(
    model_kind: str,
    train_payload: dict[str, Any],
    validation_payload: dict[str, Any] | None,
    prediction_payloads: dict[str, dict[str, Any]],
    config_payload: dict[str, Any],
    model_id: str,
    save_model: bool = True,
) -> dict[str, Any]:
    _prepare_remote_imports()
    from forecasting_models.neural_prediction import predict_neural_forecaster
    from forecasting_models.neural_training import fit_encoder_decoder_forecaster, fit_lstm_forecaster
    from modal_execution.payloads import config_from_payload, series_from_payload, series_to_payload

    train_spread = series_from_payload(train_payload)
    validation_spread = series_from_payload(validation_payload) if validation_payload else None
    config = config_from_payload(model_kind, config_payload)
    if model_kind == "lstm":
        forecaster = fit_lstm_forecaster(train_spread, validation_spread, config)
    elif model_kind == "encoder_decoder":
        forecaster = fit_encoder_decoder_forecaster(train_spread, validation_spread, config)
    else:
        raise ValueError("model_kind must be 'lstm' or 'encoder_decoder'.")

    predictions = {}
    for name, payload in prediction_payloads.items():
        spread = series_from_payload(payload)
        predictions[name] = series_to_payload(predict_neural_forecaster(forecaster, spread))

    if save_model:
        _save_forecaster(model_id, model_kind, forecaster, config_payload)
        model_volume.commit()

    return {
        "model_id": model_id,
        "model_kind": model_kind,
        "history": _history_to_payload(forecaster.history),
        "predictions": predictions,
    }


@app.function(
    gpu=GPU_FALLBACKS,
    timeout=REMOTE_TIMEOUT_SECONDS,
    volumes={str(MODEL_VOLUME_PATH): model_volume},
)
def predict_saved_spread(model_id: str, spread_payload: dict[str, Any]) -> dict[str, Any]:
    _prepare_remote_imports()
    from forecasting_models.neural_prediction import predict_neural_forecaster
    from modal_execution.payloads import series_from_payload, series_to_payload

    model_volume.reload()
    forecaster = _load_forecaster(model_id)
    spread = series_from_payload(spread_payload)
    return series_to_payload(predict_neural_forecaster(forecaster, spread))


def _prepare_remote_imports() -> None:
    remote_path = str(REMOTE_METHOD_DIR)
    if remote_path not in sys.path:
        sys.path.insert(0, remote_path)


def _save_forecaster(
    model_id: str,
    model_kind: str,
    forecaster,
    config_payload: dict[str, Any],
) -> None:
    model_dir = _model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    forecaster.model.save(str(model_dir / "model.keras"))
    metadata = {
        "model_id": model_id,
        "model_kind": model_kind,
        "model_scope": "per_pair_spread_forecaster",
        "scaler": {
            "mean": forecaster.scaler.mean,
            "scale": forecaster.scaler.scale,
        },
        "input_length": forecaster.input_length,
        "horizon": forecaster.horizon,
        "config": config_payload,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_forecaster(model_id: str):
    _prepare_remote_imports()
    from common.models import NeuralForecastModel, SeriesScaler
    from tensorflow import keras

    model_dir = _model_dir(model_id)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = keras.models.load_model(str(model_dir / "model.keras"))
    scaler = SeriesScaler(
        mean=float(metadata["scaler"]["mean"]),
        scale=float(metadata["scaler"]["scale"]),
    )
    return NeuralForecastModel(
        name=metadata["model_kind"],
        model=model,
        scaler=scaler,
        input_length=int(metadata["input_length"]),
        horizon=int(metadata["horizon"]),
    )


def _model_dir(model_id: str) -> Path:
    return MODEL_VOLUME_PATH / _safe_model_id(model_id)


def _safe_model_id(model_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in model_id)


def _history_to_payload(history) -> dict[str, list[float]]:
    if history is None:
        return {}
    return {key: [float(value) for value in values] for key, values in history.history.items()}
