"""Prediction helpers for trained neural spread forecasters."""
import numpy as np
import pandas as pd

from common.models import NeuralForecastModel
from forecasting_models.scaling import inverse_transform_array, transform_series
from forecasting_models.sequences import forecast_sequence_index, make_forecast_sequences


def predict_neural_forecaster(
    forecaster: NeuralForecastModel,
    spread: pd.Series,
) -> pd.Series:
    """Predict S*(t+horizon) indexed by the current timestamp t."""
    scaled = transform_series(spread, forecaster.scaler)
    x_pred = make_forecast_sequences(scaled.to_numpy(), forecaster.input_length)
    raw_prediction = forecaster.model.predict(x_pred, verbose=0)
    horizon_values = _last_horizon_column(raw_prediction, forecaster.horizon)
    predictions = inverse_transform_array(horizon_values, forecaster.scaler)
    index = forecast_sequence_index(spread.index, forecaster.input_length)
    return pd.Series(predictions, index=index, name="predicted_spread")


def _last_horizon_column(raw_prediction, horizon: int) -> np.ndarray:
    prediction = np.asarray(raw_prediction, dtype=float)
    if prediction.ndim == 1:
        return prediction
    column = min(horizon - 1, prediction.shape[1] - 1)
    return prediction[:, column]
