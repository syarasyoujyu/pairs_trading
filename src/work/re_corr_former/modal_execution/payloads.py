"""Serializable payload helpers for Modal ReCorrFormer execution."""
from dataclasses import fields

import numpy as np
import pandas as pd

from common.config import ModelConfig
from training.datasets import SupervisedDataset


def model_config_to_payload(config: ModelConfig) -> dict:
    """Convert a model config into a plain dictionary."""
    return vars(config).copy()


def model_config_from_payload(payload: dict) -> ModelConfig:
    """Restore a model config from a plain dictionary."""
    values = {field.name: payload[field.name] for field in fields(ModelConfig) if field.name in payload}
    return ModelConfig(**values)


def split_datasets_to_payload(splits: dict[str, SupervisedDataset]) -> dict[str, dict]:
    """Convert all dataset splits into Modal payloads."""
    return {name: dataset_to_payload(dataset) for name, dataset in splits.items()}


def split_datasets_from_payload(payload: dict[str, dict]) -> dict[str, SupervisedDataset]:
    """Restore all dataset splits from Modal payloads."""
    return {name: dataset_from_payload(dataset_payload) for name, dataset_payload in payload.items()}


def dataset_to_payload(dataset: SupervisedDataset) -> dict:
    """Convert one supervised dataset into a Modal-safe payload."""
    return {
        "inputs": {name: values for name, values in dataset.inputs.items()},
        "targets": {name: values for name, values in dataset.targets.items()},
        "metadata": frame_to_records(dataset.metadata),
    }


def dataset_from_payload(payload: dict) -> SupervisedDataset:
    """Restore one supervised dataset from a Modal-safe payload."""
    inputs = {name: np.asarray(values, dtype=np.float32) for name, values in payload["inputs"].items()}
    targets = {
        name: np.asarray(values, dtype=np.int32 if name == "trade" else np.float32)
        for name, values in payload["targets"].items()
    }
    return SupervisedDataset(inputs=inputs, targets=targets, metadata=frame_from_records(payload["metadata"]))


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a frame into records with stable date serialization."""
    payload_frame = frame.copy()
    if "date" in payload_frame.columns:
        payload_frame["date"] = pd.to_datetime(payload_frame["date"]).dt.strftime("%Y-%m-%d")
    return payload_frame.to_dict("records")


def frame_from_records(records: list[dict]) -> pd.DataFrame:
    """Restore a frame from records with parsed dates."""
    frame = pd.DataFrame(records)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame
