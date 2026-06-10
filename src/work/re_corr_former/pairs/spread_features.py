"""Spread features for candidate pairs."""
import numpy as np
import pandas as pd


def hedge_beta(log_i: pd.Series, log_j: pd.Series) -> float:
    """Estimate beta for log_i = alpha + beta * log_j."""
    x = log_j.to_numpy(dtype=float)
    y = log_i.to_numpy(dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered * x_centered))
    if denominator == 0.0:
        return 1.0
    return float(np.sum(x_centered * (y - y.mean())) / denominator)


def spread_series(log_i: pd.Series, log_j: pd.Series, beta: float) -> pd.Series:
    """Build residual spread using the supplied hedge beta."""
    intercept = float(log_i.mean() - beta * log_j.mean())
    return log_i.subtract(beta * log_j).subtract(intercept)


def spread_zscore(spread: pd.Series) -> float:
    """Return the latest z-score of a spread window."""
    scale = float(spread.std(ddof=1))
    if scale == 0.0:
        return 0.0
    return float((spread.iloc[-1] - spread.mean()) / scale)


def spread_volatility(spread: pd.Series) -> float:
    """Return spread return volatility proxy."""
    return float(spread.diff().std(ddof=1))
