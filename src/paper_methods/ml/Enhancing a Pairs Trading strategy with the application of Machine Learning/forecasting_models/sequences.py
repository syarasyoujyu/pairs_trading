"""Sequence construction for one-step and multi-step forecasting."""
import numpy as np
import pandas as pd


def make_training_sequences(
    values: np.ndarray,
    input_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create X windows and y horizons for supervised sequence learning."""
    if input_length < 1:
        raise ValueError("input_length must be at least 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    last_start = len(values) - input_length - horizon + 1
    for start in range(max(last_start, 0)):
        target_start = start + input_length
        target_end = target_start + horizon
        x_rows.append(values[start:target_start])
        y_rows.append(values[target_start:target_end])

    if not x_rows:
        raise ValueError("Not enough observations to build training sequences.")

    x = np.asarray(x_rows, dtype=float).reshape(-1, input_length, 1)
    y = np.asarray(y_rows, dtype=float)
    return x, y


def make_forecast_sequences(
    values: np.ndarray,
    input_length: int,
) -> np.ndarray:
    """Create X windows ending at each tradable timestamp."""
    if input_length < 1:
        raise ValueError("input_length must be at least 1.")
    if len(values) < input_length:
        raise ValueError("Not enough observations to build forecast sequences.")

    rows = [values[start:start + input_length] for start in range(len(values) - input_length + 1)]
    return np.asarray(rows, dtype=float).reshape(-1, input_length, 1)


def forecast_sequence_index(index: pd.Index, input_length: int) -> pd.Index:
    """Return the current-time index aligned with forecast windows."""
    if input_length < 1:
        raise ValueError("input_length must be at least 1.")
    if len(index) < input_length:
        raise ValueError("Not enough index entries for forecast windows.")
    return index[input_length - 1:]

