"""LSTM and Transformer Sharpe-ratio regression models."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common.config import SharpeModelConfig


def build_sharpe_model(
    sequence_shape: tuple[int, int],
    scalar_count: int,
    config: SharpeModelConfig,
    seed: int,
) -> keras.Model:
    """Build the configured Sharpe-ratio regression model."""
    keras.utils.set_random_seed(seed)
    if config.model_type == "lstm":
        return build_lstm_sharpe_model(sequence_shape, scalar_count, config)
    if config.model_type == "transformer":
        return build_transformer_sharpe_model(sequence_shape, scalar_count, config)
    raise ValueError(f"Unsupported sharpe model_type: {config.model_type}")


def build_lstm_sharpe_model(
    sequence_shape: tuple[int, int],
    scalar_count: int,
    config: SharpeModelConfig,
) -> keras.Model:
    """Build a shared-LSTM pair Sharpe model."""
    asset_i_input, asset_j_input, scalar_input = sharpe_inputs(sequence_shape, scalar_count)
    encoder = layers.LSTM(config.encoder_units, name="shared_ohlcv_lstm")
    z_i = encoder(asset_i_input)
    z_j = encoder(asset_j_input)
    output = sharpe_output(z_i, z_j, scalar_input, config)
    return compile_sharpe_model(asset_i_input, asset_j_input, scalar_input, output, config, "SharpeLSTM")


def build_transformer_sharpe_model(
    sequence_shape: tuple[int, int],
    scalar_count: int,
    config: SharpeModelConfig,
) -> keras.Model:
    """Build a shared-Transformer pair Sharpe model."""
    asset_i_input, asset_j_input, scalar_input = sharpe_inputs(sequence_shape, scalar_count)
    encoder = build_shared_transformer_encoder(sequence_shape, config)
    z_i = encoder(asset_i_input)
    z_j = encoder(asset_j_input)
    output = sharpe_output(z_i, z_j, scalar_input, config)
    return compile_sharpe_model(asset_i_input, asset_j_input, scalar_input, output, config, "SharpeTransformer")


def sharpe_inputs(sequence_shape: tuple[int, int], scalar_count: int):
    """Return model input layers."""
    asset_i_input = keras.Input(shape=sequence_shape, name="asset_i_ohlcv")
    asset_j_input = keras.Input(shape=sequence_shape, name="asset_j_ohlcv")
    scalar_input = keras.Input(shape=(scalar_count,), name="pair_scalars")
    return asset_i_input, asset_j_input, scalar_input


def build_shared_transformer_encoder(sequence_shape: tuple[int, int], config: SharpeModelConfig) -> keras.Model:
    """Build the shared OHLCV Transformer encoder."""
    inputs = keras.Input(shape=sequence_shape, name="ohlcv_sequence")
    hidden = layers.Dense(config.encoder_units, name="input_projection")(inputs)
    for block_index in range(max(1, config.transformer_layers)):
        hidden = transformer_block(hidden, config, block_index + 1)
    pooled = layers.GlobalAveragePooling1D(name="sequence_pooling")(hidden)
    return keras.Model(inputs, pooled, name="shared_ohlcv_transformer")


def transformer_block(hidden, config: SharpeModelConfig, block_number: int):
    """Apply one Transformer encoder block."""
    key_dim = max(1, config.encoder_units // max(1, config.transformer_heads))
    attention = layers.MultiHeadAttention(
        num_heads=config.transformer_heads,
        key_dim=key_dim,
        dropout=config.dropout,
        name=f"self_attention_{block_number}",
    )(hidden, hidden)
    attention = layers.Dropout(config.dropout, name=f"attention_dropout_{block_number}")(attention)
    hidden = layers.LayerNormalization(name=f"attention_norm_{block_number}")(hidden + attention)
    feed_forward = layers.Dense(config.transformer_ff_dim, activation="relu", name=f"ffn_up_{block_number}")(hidden)
    feed_forward = layers.Dropout(config.dropout, name=f"ffn_dropout_{block_number}")(feed_forward)
    feed_forward = layers.Dense(config.encoder_units, name=f"ffn_down_{block_number}")(feed_forward)
    return layers.LayerNormalization(name=f"ffn_norm_{block_number}")(hidden + feed_forward)


def sharpe_output(z_i, z_j, scalar_input, config: SharpeModelConfig):
    """Return the Sharpe regression output from pair embeddings."""
    abs_diff = layers.Lambda(lambda values: tf.abs(values[0] - values[1]), name="abs_embedding_diff")([z_i, z_j])
    product = layers.Multiply(name="embedding_product")([z_i, z_j])
    pair_features = layers.Concatenate(name="pair_representation")([z_i, z_j, abs_diff, product, scalar_input])
    hidden = pair_dense_stack(pair_features, config)
    return layers.Dense(1, activation="linear", name="predicted_sharpe")(hidden)


def pair_dense_stack(pair_features, config: SharpeModelConfig):
    """Apply the configurable dense stack after pair representation."""
    hidden = pair_features
    for layer_index in range(max(1, config.dense_layers)):
        layer_number = layer_index + 1
        hidden = layers.Dense(config.dense_units, activation="relu", name=f"pair_dense_{layer_number}")(hidden)
        hidden = layers.Dropout(config.dropout, name=f"pair_dropout_{layer_number}")(hidden)
    return hidden


def compile_sharpe_model(asset_i_input, asset_j_input, scalar_input, output, config: SharpeModelConfig, name: str) -> keras.Model:
    """Compile a Sharpe model."""
    model = keras.Model(
        inputs={
            "asset_i_ohlcv": asset_i_input,
            "asset_j_ohlcv": asset_j_input,
            "pair_scalars": scalar_input,
        },
        outputs=output,
        name=name,
    )
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model
