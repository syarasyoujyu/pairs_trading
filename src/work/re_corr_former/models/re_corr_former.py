"""Keras implementation of the ReCorrFormer pair scoring model."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common.config import ModelConfig


def build_re_corr_former(
    sequence_shape: tuple[int, int],
    scalar_count: int,
    config: ModelConfig,
) -> keras.Model:
    """Build a pair scoring network whose only target is future correlation recovery."""
    asset_i_input = keras.Input(shape=sequence_shape, name="asset_i_sequence")
    asset_j_input = keras.Input(shape=sequence_shape, name="asset_j_sequence")
    scalar_input = keras.Input(shape=(scalar_count,), name="pair_scalars")

    encoder = layers.LSTM(config.encoder_units, name="shared_asset_lstm")
    z_i = encoder(asset_i_input)
    z_j = encoder(asset_j_input)
    abs_diff = layers.Lambda(lambda values: tf.abs(values[0] - values[1]), name="abs_embedding_diff")([z_i, z_j])
    product = layers.Multiply(name="embedding_product")([z_i, z_j])
    pair_features = layers.Concatenate(name="pair_representation")([z_i, z_j, abs_diff, product, scalar_input])
    hidden = pair_dense_stack(pair_features, config)

    corr = layers.Dense(1, activation="linear", name="corr")(hidden)

    model = keras.Model(
        inputs={
            "asset_i_sequence": asset_i_input,
            "asset_j_sequence": asset_j_input,
            "pair_scalars": scalar_input,
        },
        outputs={"corr": corr},
        name="ReCorrFormer",
    )
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={"corr": "mse"},
        metrics={"corr": ["mae"]},
    )
    return model


def pair_dense_stack(pair_features, config: ModelConfig):
    """Apply the configurable pair-scoring dense stack."""
    hidden = pair_features
    for layer_index in range(max(1, config.dense_layers)):
        layer_number = layer_index + 1
        hidden = layers.Dense(config.dense_units, activation="relu", name=f"pair_dense_{layer_number}")(hidden)
        if layer_index == 0:
            hidden = layers.Dropout(config.dropout, name="pair_dropout_1")(hidden)
    return hidden


def set_tensorflow_seed(seed: int) -> None:
    """Set TensorFlow's random seed."""
    tf.random.set_seed(seed)
