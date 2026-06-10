"""Feature construction: normalized returns followed by PCA."""
import numpy as np
import pandas as pd


def normalized_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute Ri,t = (Pi,t - Pi,t-1) / Pi,t-1 for every asset."""
    returns = prices.astype(float).pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.dropna(axis=0, how="any")


def pca_features(
    returns: pd.DataFrame,
    n_components: int = 5,
    max_components: int = 15,
    standardize: bool = True,
) -> pd.DataFrame:
    """Represent each asset by the first PCA scores of its return path."""
    if returns.empty:
        raise ValueError("PCA requires a non-empty return matrix.")
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")
    if n_components > max_components:
        raise ValueError(f"The paper caps PCA dimensions at {max_components}.")

    x = returns.T.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)

    if standardize:
        scale = x.std(axis=0, ddof=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        x = x / scale

    max_rank = min(x.shape)
    n_used = min(n_components, max_rank)
    u, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    scores = u[:, :n_used] * singular_values[:n_used]
    columns = [f"pc{i + 1}" for i in range(n_used)]
    return pd.DataFrame(scores, index=returns.columns, columns=columns)

