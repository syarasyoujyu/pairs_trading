"""Modal app for ISEPT CAE and rolling MLP training."""
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

import modal


APP_NAME = "isept-image-pair-trading"
LOCAL_METHOD_DIR = Path(__file__).resolve().parents[1]
REMOTE_METHOD_DIR = Path("/root/isept")
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
    .uv_pip_install("pandas>=3.0.3", "tensorflow[and-cuda]>=2.21.0")
    .env({"LD_LIBRARY_PATH": ":".join(CUDA_LIBRARY_PATHS)})
    .add_local_dir(LOCAL_METHOD_DIR, remote_path=str(REMOTE_METHOD_DIR), ignore=["**/__pycache__/**", "*.pyc"])
)
app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU_FALLBACKS, timeout=REMOTE_TIMEOUT_SECONDS)
def check_tensorflow_gpu() -> dict[str, Any]:
    """Return TensorFlow GPU visibility inside Modal."""
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    return {
        "tensorflow_version": tf.__version__,
        "gpu_devices": [device.name for device in devices],
        "gpu_count": len(devices),
    }


@app.function(gpu=GPU_FALLBACKS, timeout=REMOTE_TIMEOUT_SECONDS)
def train_and_select_isept_pairs(
    images,
    image_metadata: list[dict],
    label_records: list[dict],
    config_payload: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Train CAE plus rolling MLP on Modal and return selected pairs."""
    _prepare_remote_imports()
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)
    image_array = np.asarray(images, dtype=np.float32)
    encoder, autoencoder = build_cae_model(image_array.shape[1:], config_payload, keras, layers)
    train_x, validation_x = split_images(image_array, seed)
    history = autoencoder.fit(
        train_x,
        train_x,
        validation_data=(validation_x, validation_x),
        epochs=int(config_payload["cae_epochs"]),
        batch_size=int(config_payload["batch_size"]),
        verbose=1,
        shuffle=True,
        callbacks=cae_callbacks(config_payload, keras),
    )
    raw_embeddings = encoder.predict(image_array, batch_size=int(config_payload["batch_size"]), verbose=0)
    embedding_frame = aggregate_embeddings(raw_embeddings, image_metadata)
    label_frame = pd.DataFrame(label_records)
    selected = rolling_mlp_selection(embedding_frame, label_frame, config_payload, seed, keras, layers)
    return {
        "selected_pairs": selected.to_dict("records"),
        "cae_history": history_to_payload(history),
        "embedding_count": int(len(embedding_frame)),
    }


def _prepare_remote_imports() -> None:
    remote_path = str(REMOTE_METHOD_DIR)
    if remote_path not in sys.path:
        sys.path.insert(0, remote_path)


def build_cae_model(input_shape, config: dict[str, Any], keras, layers):
    """Build the ISEPT-style convolutional autoencoder."""
    base_filters = int(config["cae_base_filters"])
    inputs = keras.Input(shape=input_shape, name="candlestick_image")
    x = layers.Conv2D(base_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(base_filters * 2, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(base_filters * 4, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    latent = layers.Flatten(name="stock_latent")(x)
    encoder = keras.Model(inputs, latent, name="isept_cae_encoder")

    latent_height = int(input_shape[0]) // 8
    latent_width = int(input_shape[1]) // 8
    latent_dim = latent_height * latent_width * base_filters * 4
    decoder_input = keras.Input(shape=(latent_dim,), name="latent_input")
    x = layers.Reshape((latent_height, latent_width, base_filters * 4))(decoder_input)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(base_filters * 2, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(base_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_input, decoded, name="isept_cae_decoder")
    outputs = decoder(encoder(inputs))
    autoencoder = keras.Model(inputs, outputs, name="isept_cae")
    optimizer = keras.optimizers.Adam(learning_rate=float(config["cae_learning_rate"]))
    autoencoder.compile(optimizer=optimizer, loss="mse")
    return encoder, autoencoder


def cae_callbacks(config: dict[str, Any], keras) -> list[Any]:
    """Return paper-style CAE callbacks."""
    return [
        keras.callbacks.LearningRateScheduler(lambda epoch, lr: cae_learning_rate(epoch, lr, config)),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config["cae_patience"]),
            restore_best_weights=True,
        ),
    ]


def cae_learning_rate(epoch: int, current_lr: float, config: dict[str, Any]) -> float:
    """Halve the CAE learning rate every configured number of epochs."""
    decay_epochs = max(1, int(config["cae_lr_decay_epochs"]))
    if epoch > 0 and epoch % decay_epochs == 0:
        return float(current_lr) * float(config["cae_lr_decay_factor"])
    return float(current_lr)


def split_images(images, seed: int):
    """Split image tensors into train and validation sets."""
    import numpy as np

    count = len(images)
    if count < 2:
        raise ValueError("At least two images are required for CAE train/validation split.")
    order = np.random.default_rng(seed).permutation(count)
    shuffled = images[order]
    split = max(1, int(count * 0.70))
    if split >= count:
        split = count - 1
    return shuffled[:split], shuffled[split:]


def aggregate_embeddings(raw_embeddings, metadata: list[dict]):
    """Average image embeddings by month and symbol."""
    import pandas as pd

    frame = pd.DataFrame(metadata)
    embedding_columns = [f"z_{column}" for column in range(raw_embeddings.shape[1])]
    embeddings = pd.DataFrame(raw_embeddings, columns=embedding_columns)
    joined = pd.concat([frame.reset_index(drop=True), embeddings], axis=1)
    return joined.groupby(["month_end", "symbol"], as_index=False)[embedding_columns].mean()


def rolling_mlp_selection(embedding_frame, label_frame, config: dict[str, Any], seed: int, keras, layers):
    """Train a rolling pair MLP and select top predicted-Sharpe pairs."""
    import numpy as np
    import pandas as pd

    selected_rows: list[dict] = []
    months = sorted(embedding_frame["month_end"].unique())
    for month_index, month in enumerate(months):
        if month_index < int(config["warmup_months"]):
            continue
        training_rows = feedback_training_rows(label_frame[label_frame["label_month"] < month], config)
        if training_rows.empty:
            continue
        x_train, y_train = pair_training_arrays(training_rows, embedding_frame)
        pca_mean, pca_components = fit_pca(x_train, int(config["pca_components"]))
        x_train = apply_pca(x_train, pca_mean, pca_components)
        x_fit, x_validation, y_fit, y_validation = split_supervised_arrays(x_train, y_train, seed + month_index)
        model = build_pair_mlp(x_train.shape[1], config, seed + month_index, keras, layers)
        validation_data = (x_validation, y_validation) if len(x_validation) else None
        model.fit(
            x_fit,
            y_fit,
            epochs=int(config["mlp_epochs"]),
            batch_size=int(config["batch_size"]),
            verbose=0,
            shuffle=True,
            validation_data=validation_data,
            callbacks=mlp_callbacks(config, keras) if validation_data else None,
        )
        candidate_frame, x_candidates = current_month_pair_candidates(embedding_frame, month)
        if candidate_frame.empty:
            continue
        x_candidates = apply_pca(x_candidates, pca_mean, pca_components)
        candidate_frame["predicted_sharpe"] = model.predict(x_candidates, verbose=0).reshape(-1)
        selected_rows.extend(select_top_pairs(candidate_frame, config))
    return pd.DataFrame(selected_rows)


def feedback_training_rows(label_frame, config: dict[str, Any]):
    """Keep top/bottom realized-Sharpe labels for every feedback month."""
    import pandas as pd

    rows = []
    per_side = int(config["feedback_pairs_per_side"])
    for _, group in label_frame.groupby("label_month", sort=True):
        rows.append(group.nlargest(per_side, "realized_sharpe"))
        rows.append(group.nsmallest(per_side, "realized_sharpe"))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True).drop_duplicates(["label_month", "asset_i", "asset_j"])


def pair_training_arrays(training_rows, embedding_frame):
    """Build MLP training arrays from labeled pairs and embeddings."""
    import numpy as np

    x_rows = []
    y_rows = []
    for _, row in training_rows.iterrows():
        x_rows.append(pair_embedding(embedding_frame, row["label_month"], row["asset_i"], row["asset_j"]))
        y_rows.append(float(row["realized_sharpe"]))
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def fit_pca(features, component_count: int):
    """Fit a compact PCA projection with NumPy SVD."""
    import numpy as np

    mean = features.mean(axis=0).astype(np.float32)
    centered = features.astype(np.float32) - mean
    components_to_keep = min(max(1, component_count), centered.shape[0], centered.shape[1])
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return mean, vectors[:components_to_keep].astype(np.float32)


def apply_pca(features, mean, components):
    """Project pair embeddings through a fitted PCA basis."""
    return (features.astype("float32") - mean).dot(components.T).astype("float32")


def split_supervised_arrays(features, targets, seed: int):
    """Split MLP training data into train and validation arrays."""
    import numpy as np

    count = len(features)
    if count < 2:
        return features, np.empty((0, features.shape[1]), dtype=np.float32), targets, np.empty((0,), dtype=np.float32)
    order = np.random.default_rng(seed).permutation(count)
    features = features[order]
    targets = targets[order]
    split = max(1, int(count * 0.70))
    if split >= count:
        split = count - 1
    return features[:split], features[split:], targets[:split], targets[split:]


def current_month_pair_candidates(embedding_frame, month: str):
    """Build pair candidates and feature matrix for one month."""
    import numpy as np
    import pandas as pd

    month_frame = embedding_frame[embedding_frame["month_end"] == month]
    symbols = sorted(month_frame["symbol"].astype(str).tolist())
    rows = []
    x_rows = []
    for asset_i, asset_j in combinations(symbols, 2):
        rows.append({"month_end": month, "asset_i": asset_i, "asset_j": asset_j})
        x_rows.append(pair_embedding(embedding_frame, month, asset_i, asset_j))
    return pd.DataFrame(rows), np.asarray(x_rows, dtype=np.float32)


def pair_embedding(embedding_frame, month: str, asset_i: str, asset_j: str):
    """Return concatenated pair embedding."""
    left = symbol_embedding(embedding_frame, month, asset_i)
    right = symbol_embedding(embedding_frame, month, asset_j)
    return left + right


def symbol_embedding(embedding_frame, month: str, symbol: str) -> list[float]:
    """Return one stock embedding vector."""
    row = embedding_frame[(embedding_frame["month_end"] == month) & (embedding_frame["symbol"] == symbol)].iloc[0]
    columns = [column for column in embedding_frame.columns if column.startswith("z_")]
    return [float(row[column]) for column in columns]


def build_pair_mlp(input_dim: int, config: dict[str, Any], seed: int, keras, layers):
    """Build the pair Sharpe regression MLP."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.LayerNormalization(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.50),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.50),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.50),
            layers.Dense(1),
        ],
        name="isept_pair_mlp",
    )
    optimizer = keras.optimizers.Adam(learning_rate=float(config["mlp_learning_rate"]))
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def mlp_callbacks(config: dict[str, Any], keras) -> list[Any]:
    """Return paper-style MLP callbacks."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config["mlp_patience"]),
            restore_best_weights=True,
        )
    ]


def select_top_pairs(candidate_frame, config: dict[str, Any]) -> list[dict]:
    """Select top predicted-Sharpe pairs with optional no-overlap matching."""
    ranked = candidate_frame.sort_values("predicted_sharpe", ascending=False)
    used_assets: set[str] = set()
    rows = []
    for _, row in ranked.iterrows():
        asset_i = str(row["asset_i"])
        asset_j = str(row["asset_j"])
        if not bool(config["allow_asset_reuse"]) and (asset_i in used_assets or asset_j in used_assets):
            continue
        rows.append(
            {
                "month_end": row["month_end"],
                "asset_i": asset_i,
                "asset_j": asset_j,
                "predicted_sharpe": float(row["predicted_sharpe"]),
                "rank": len(rows) + 1,
            }
        )
        used_assets.update([asset_i, asset_j])
        if len(rows) >= int(config["top_k_pairs"]):
            break
    return rows


def history_to_payload(history) -> dict[str, list[float]]:
    """Serialize Keras history."""
    return {key: [float(value) for value in values] for key, values in history.history.items()}
