"""Threshold construction for standard and forecasting-based models."""
import pandas as pd

from common.models import ForecastThresholds, StandardThresholds
from pair_selection.spreads import spread_percentage_change


def standard_thresholds(spread: pd.Series, entry_sigma: float = 2.0) -> StandardThresholds:
    """Compute Gatev-style long, short, and exit thresholds."""
    mean = float(spread.mean())
    std = float(spread.std(ddof=1))
    return StandardThresholds(
        long_entry=mean - entry_sigma * std,
        short_entry=mean + entry_sigma * std,
        exit=mean,
    )


def forecast_threshold_candidates(
    spread: pd.Series,
    horizon: int = 1,
) -> tuple[ForecastThresholds, ForecastThresholds]:
    """Build quintile and decile threshold candidates from spread changes."""
    changes = spread_percentage_change(spread, horizon=horizon)
    positive = changes[changes > 0.0]
    negative = changes[changes < 0.0]
    if positive.empty or negative.empty:
        raise ValueError("Forecast thresholds require positive and negative spread changes.")

    quintile = ForecastThresholds(
        name="quintile",
        short_entry=float(negative.quantile(0.20)),
        long_entry=float(positive.quantile(0.80)),
    )
    decile = ForecastThresholds(
        name="decile",
        short_entry=float(negative.quantile(0.10)),
        long_entry=float(positive.quantile(0.90)),
    )
    return quintile, decile
