"""Spread construction and spread percentage changes."""
import numpy as np
import pandas as pd


def fit_hedge_ratio(dependent: pd.Series, independent: pd.Series) -> tuple[float, float]:
    """Fit dependent = intercept + hedge_ratio * independent by OLS."""
    aligned = pd.concat([dependent, independent], axis=1).dropna()
    if len(aligned) < 3:
        raise ValueError("At least three aligned observations are required.")

    y = aligned.iloc[:, 0].to_numpy(dtype=float)
    x = aligned.iloc[:, 1].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    intercept, hedge_ratio = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(hedge_ratio), float(intercept)


def build_spread(
    dependent: pd.Series,
    independent: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
) -> pd.Series:
    """Build St = Yt - intercept - beta * Xt."""
    aligned = pd.concat([dependent, independent], axis=1).dropna()
    values = aligned.iloc[:, 0] - intercept - hedge_ratio * aligned.iloc[:, 1]
    return pd.Series(values.to_numpy(dtype=float), index=aligned.index, name="spread")


def spread_percentage_change(spread: pd.Series, horizon: int = 1) -> pd.Series:
    """Compute xt = (S(t+horizon) - S(t)) / S(t) * 100."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    values = spread.astype(float)
    future = values.shift(-horizon)
    denominator = values.replace(0.0, np.nan)
    change = (future - values) / denominator * 100.0
    return change.replace([np.inf, -np.inf], np.nan).dropna()

