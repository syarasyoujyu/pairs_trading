"""Training routines for neural spread forecasters."""
import pandas as pd

from common.models import EncoderDecoderForecastConfig, LSTMForecastConfig, NeuralForecastModel
from forecasting_models.neural_models import build_encoder_decoder_model, build_lstm_model, early_stopping_callback
from forecasting_models.scaling import fit_series_scaler, transform_series
from forecasting_models.sequences import make_training_sequences


def fit_lstm_forecaster(
    train_spread: pd.Series,
    validation_spread: pd.Series | None,
    config: LSTMForecastConfig,
) -> NeuralForecastModel:
    """Train a direct LSTM forecaster on the formation spread."""
    scaler = fit_series_scaler(train_spread)
    x_train, y_train = _scaled_training_data(train_spread, scaler, config.input_length, config.horizon)
    validation_data = _scaled_validation_data(validation_spread, scaler, config.input_length, config.horizon)
    model = build_lstm_model(config)
    callbacks = [early_stopping_callback(config.patience)] if validation_data is not None else []
    history = model.fit(
        x_train,
        y_train,
        validation_data=validation_data,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )
    return NeuralForecastModel("lstm", model, scaler, config.input_length, config.horizon, history)


def fit_encoder_decoder_forecaster(
    train_spread: pd.Series,
    validation_spread: pd.Series | None,
    config: EncoderDecoderForecastConfig,
) -> NeuralForecastModel:
    """Train an LSTM encoder-decoder forecaster on the formation spread."""
    scaler = fit_series_scaler(train_spread)
    x_train, y_train = _scaled_training_data(train_spread, scaler, config.input_length, config.horizon)
    validation_data = _scaled_validation_data(validation_spread, scaler, config.input_length, config.horizon)
    model = build_encoder_decoder_model(config)
    callbacks = [early_stopping_callback(config.patience)] if validation_data is not None else []
    history = model.fit(
        x_train,
        y_train,
        validation_data=validation_data,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )
    return NeuralForecastModel("encoder_decoder", model, scaler, config.input_length, config.horizon, history)


def _scaled_training_data(
    spread: pd.Series,
    scaler,
    input_length: int,
    horizon: int,
):
    scaled = transform_series(spread, scaler).to_numpy()
    return make_training_sequences(scaled, input_length, horizon)


def _scaled_validation_data(
    spread: pd.Series | None,
    scaler,
    input_length: int,
    horizon: int,
):
    if spread is None or spread.empty:
        return None
    if len(spread.dropna()) < input_length + horizon:
        return None
    x_val, y_val = _scaled_training_data(spread, scaler, input_length, horizon)
    return x_val, y_val
