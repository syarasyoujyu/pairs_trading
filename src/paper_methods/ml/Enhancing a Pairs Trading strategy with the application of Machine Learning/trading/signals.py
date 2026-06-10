"""Signal and trade generation for the two trading models."""
import numpy as np
import pandas as pd

from common.models import ForecastThresholds, StandardThresholds
from trading.thresholds import standard_thresholds


def generate_standard_signals(
    spread: pd.Series,
    thresholds: StandardThresholds | None = None,
) -> pd.DataFrame:
    """Generate Gatev-style threshold signals from a spread series."""
    levels = thresholds or standard_thresholds(spread)
    rows: list[dict] = []
    position = 0

    for timestamp, value in spread.astype(float).items():
        if position == 0 and value <= levels.long_entry:
            position = 1
        elif position == 0 and value >= levels.short_entry:
            position = -1
        elif position == 1 and value >= levels.exit:
            position = 0
        elif position == -1 and value <= levels.exit:
            position = 0

        rows.append(
            {
                "datetime": timestamp,
                "spread": float(value),
                "signal": position,
                "long_entry": levels.long_entry,
                "short_entry": levels.short_entry,
                "exit": levels.exit,
            }
        )

    return pd.DataFrame(rows)


def generate_forecast_signals(
    spread: pd.Series,
    predicted_spread: pd.Series,
    thresholds: ForecastThresholds,
) -> pd.DataFrame:
    """Generate signals from predicted spread percentage changes."""
    aligned = pd.concat(
        [spread.astype(float).rename("spread"), predicted_spread.astype(float).rename("predicted_spread")],
        axis=1,
    ).dropna()
    rows: list[dict] = []
    position = 0

    for timestamp, row in aligned.iterrows():
        value = float(row["spread"])
        predicted = float(row["predicted_spread"])
        if value == 0.0:
            predicted_change = np.nan
        else:
            predicted_change = (predicted - value) / value * 100.0

        if np.isfinite(predicted_change):
            if position == 0 and predicted_change >= thresholds.long_entry:
                position = 1
            elif position == 0 and predicted_change <= thresholds.short_entry:
                position = -1
            elif position == 1 and predicted_change <= 0.0:
                position = 0
            elif position == -1 and predicted_change >= 0.0:
                position = 0

        rows.append(
            {
                "datetime": timestamp,
                "spread": value,
                "predicted_spread": predicted,
                "predicted_change_pct": float(predicted_change),
                "signal": position,
                "threshold_set": thresholds.name,
                "long_entry_pct": thresholds.long_entry,
                "short_entry_pct": thresholds.short_entry,
            }
        )

    return pd.DataFrame(rows)


def spread_strategy_return(spread: pd.Series, signals: pd.Series) -> float:
    """Estimate validation return as held signal times next spread change."""
    aligned = pd.concat([spread.astype(float), signals.astype(float)], axis=1).dropna()
    if aligned.empty:
        return 0.0
    spread_values = aligned.iloc[:, 0]
    signal_values = aligned.iloc[:, 1]
    profit = signal_values.shift(1).fillna(0.0) * spread_values.diff().fillna(0.0)
    return float(profit.sum())


def signals_to_trades(
    signals_df: pd.DataFrame,
    asset_y: str,
    asset_x: str,
    hedge_ratio: float,
) -> pd.DataFrame:
    """Convert signal transitions into open and close trade records."""
    trades: list[dict] = []
    current = 0

    for _, row in signals_df.iterrows():
        new_position = int(row["signal"])
        if new_position == current:
            continue

        if current != 0:
            trades.append(
                {
                    "datetime": row["datetime"],
                    "action": "CLOSE",
                    "direction": "LONG" if current > 0 else "SHORT",
                    f"{asset_y}_side": "SELL" if current > 0 else "BUY",
                    f"{asset_x}_side": "BUY" if current > 0 else "SELL",
                    f"{asset_y}_lots": 1.0,
                    f"{asset_x}_lots": round(abs(hedge_ratio), 4),
                    "spread": row["spread"],
                    "signal": new_position,
                }
            )

        if new_position != 0:
            trades.append(
                {
                    "datetime": row["datetime"],
                    "action": "OPEN",
                    "direction": "LONG" if new_position > 0 else "SHORT",
                    f"{asset_y}_side": "BUY" if new_position > 0 else "SELL",
                    f"{asset_x}_side": "SELL" if new_position > 0 else "BUY",
                    f"{asset_y}_lots": 1.0,
                    f"{asset_x}_lots": round(abs(hedge_ratio), 4),
                    "spread": row["spread"],
                    "signal": new_position,
                }
            )

        current = new_position

    return pd.DataFrame(trades)
