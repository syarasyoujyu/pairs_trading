"""Serializable payload helpers for Modal neural forecasting."""
from dataclasses import asdict
from typing import Any

import pandas as pd

from common.models import EncoderDecoderForecastConfig, LSTMForecastConfig


def series_to_payload(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna().astype(float)
    return {
        "index": [timestamp.isoformat() for timestamp in pd.to_datetime(clean.index)],
        "values": clean.tolist(),
        "name": clean.name,
    }


def series_from_payload(payload: dict[str, Any]) -> pd.Series:
    index = pd.to_datetime(payload["index"])
    return pd.Series(payload["values"], index=index, name=payload.get("name"), dtype=float)


def config_to_payload(config: LSTMForecastConfig | EncoderDecoderForecastConfig) -> dict[str, Any]:
    return asdict(config)


def config_from_payload(
    model_kind: str,
    payload: dict[str, Any],
) -> LSTMForecastConfig | EncoderDecoderForecastConfig:
    if model_kind == "lstm":
        return LSTMForecastConfig(**payload)
    if model_kind == "encoder_decoder":
        return EncoderDecoderForecastConfig(**payload)
    raise ValueError("model_kind must be 'lstm' or 'encoder_decoder'.")
