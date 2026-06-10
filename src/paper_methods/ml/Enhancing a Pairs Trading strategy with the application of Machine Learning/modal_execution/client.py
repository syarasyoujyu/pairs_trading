"""Local client wrappers for Modal neural forecasting."""
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import modal
import pandas as pd

from common.models import EncoderDecoderForecastConfig, LSTMForecastConfig
from modal_execution.app import app, check_tensorflow_gpu, predict_saved_spread, train_and_predict_spread
from modal_execution.payloads import config_to_payload, series_from_payload, series_to_payload


@contextmanager
def modal_app_context() -> Iterator[None]:
    with modal.enable_output():
        with app.run():
            yield


def modal_gpu_status() -> dict[str, Any]:
    return check_tensorflow_gpu.remote()


def train_and_predict_neural_forecaster(
    model_kind: str,
    train_spread: pd.Series,
    validation_spread: pd.Series | None,
    prediction_spreads: dict[str, pd.Series],
    config: LSTMForecastConfig | EncoderDecoderForecastConfig,
    model_id: str,
    save_model: bool = True,
) -> dict[str, Any]:
    result = train_and_predict_spread.remote(
        model_kind,
        series_to_payload(train_spread),
        series_to_payload(validation_spread) if validation_spread is not None else None,
        {name: series_to_payload(spread) for name, spread in prediction_spreads.items()},
        config_to_payload(config),
        model_id,
        save_model,
    )
    return {
        **result,
        "predictions": {
            name: series_from_payload(payload)
            for name, payload in result["predictions"].items()
        },
    }


def predict_saved_neural_forecaster(model_id: str, spread: pd.Series) -> pd.Series:
    payload = predict_saved_spread.remote(model_id, series_to_payload(spread))
    return series_from_payload(payload)
