"""Keras model builders for LSTM forecasting."""
from tensorflow import keras

from common.models import EncoderDecoderForecastConfig, LSTMForecastConfig


def build_lstm_model(config: LSTMForecastConfig):
    """Build the paper's direct LSTM spread forecaster."""
    model = keras.Sequential(name="lstm_spread_forecaster")
    model.add(keras.layers.Input(shape=(config.input_length, 1)))
    for layer_id in range(config.hidden_layers):
        return_sequences = layer_id < config.hidden_layers - 1
        model.add(keras.layers.LSTM(config.hidden_units, return_sequences=return_sequences))
        if config.dropout > 0.0:
            model.add(keras.layers.Dropout(config.dropout))
    model.add(keras.layers.Dense(config.horizon))
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def build_encoder_decoder_model(config: EncoderDecoderForecastConfig):
    """Build an LSTM encoder-decoder for multi-step spread forecasts."""
    model = keras.Sequential(name="lstm_encoder_decoder_spread_forecaster")
    model.add(keras.layers.Input(shape=(config.input_length, 1)))
    model.add(keras.layers.LSTM(config.encoder_units))
    if config.dropout > 0.0:
        model.add(keras.layers.Dropout(config.dropout))
    model.add(keras.layers.RepeatVector(config.horizon))
    model.add(keras.layers.LSTM(config.decoder_units, return_sequences=True))
    if config.dropout > 0.0:
        model.add(keras.layers.Dropout(config.dropout))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
    model.add(keras.layers.Reshape((config.horizon,)))
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def early_stopping_callback(patience: int):
    """Create a restore-best-weights early stopping callback."""
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
