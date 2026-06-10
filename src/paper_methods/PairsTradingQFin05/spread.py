"""
Spread construction for FX pairs.

The observed spread y_k is modelled as:
    y_k = log(price1_k) - β * log(price2_k)

where β (hedge ratio) is estimated via OLS so the spread is stationary.
"""
import numpy as np
import pandas as pd


def compute_hedge_ratio(log_p1: np.ndarray, log_p2: np.ndarray) -> float:
    """OLS hedge ratio β: log_p1 = β * log_p2 + α."""
    cov = np.cov(log_p1, log_p2)
    return float(cov[0, 1] / cov[1, 1])


def build_spread(
    price1: pd.Series,
    price2: pd.Series,
    hedge_ratio: float | None = None,
) -> tuple[np.ndarray, float]:
    """Build log spread y_k = log(price1_k) - β * log(price2_k).

    Args:
        price1:       Close prices of the first pair.
        price2:       Close prices of the second pair (same index).
        hedge_ratio:  Fixed β; estimated via OLS if None.

    Returns:
        (spread array, hedge_ratio β)
    """
    log1 = np.log(price1.values.astype(float))
    log2 = np.log(price2.values.astype(float))

    if hedge_ratio is None:
        hedge_ratio = compute_hedge_ratio(log1, log2)

    spread = log1 - hedge_ratio * log2
    return spread, hedge_ratio
