"""Validation utilities for choosing forecast thresholds."""
import pandas as pd

from common.models import ForecastThresholds
from trading.signals import generate_forecast_signals, spread_strategy_return


def choose_forecast_thresholds(
    spread: pd.Series,
    predicted_spread: pd.Series,
    candidates: tuple[ForecastThresholds, ...],
) -> ForecastThresholds:
    """Choose the candidate with the highest validation spread return."""
    if not candidates:
        raise ValueError("At least one threshold candidate is required.")

    best_thresholds = candidates[0]
    best_return = float("-inf")
    for thresholds in candidates:
        signals = generate_forecast_signals(spread, predicted_spread, thresholds)
        score = spread_strategy_return(
            signals.set_index("datetime")["spread"],
            signals.set_index("datetime")["signal"],
        )
        if score > best_return:
            best_return = score
            best_thresholds = thresholds
    return best_thresholds
