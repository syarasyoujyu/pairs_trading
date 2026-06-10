"""Vidyamurthy-style pair trading execution for ISEPT-selected pairs."""
import pandas as pd

from common.config import TradingConfig
from trading.gatev import build_log_spread, fit_log_spread, signal_returns, trade_records_from_signals


def simulate_vidyamurthy_pair(
    history_prices: pd.DataFrame,
    trading_prices: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    config: TradingConfig,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Simulate one pair with Vidyamurthy's opposite-band spread rule."""
    gamma, intercept, mean, std = fit_log_spread(history_prices, asset_i, asset_j)
    spread = build_log_spread(trading_prices, asset_i, asset_j, gamma, intercept)
    signals = vidyamurthy_signals(spread, mean, std, config)
    returns = signal_returns(trading_prices, signals, asset_i, asset_j, gamma, config)
    trades = trade_records_from_signals(signals, returns, asset_i, asset_j)
    signals["asset_i"] = asset_i
    signals["asset_j"] = asset_j
    signals["gamma"] = gamma
    signals["intercept"] = intercept
    signals["spread_mean"] = mean
    signals["spread_std"] = std
    signals["trade_rule"] = "vidyamurthy"
    return returns, signals, trades


def vidyamurthy_signals(spread: pd.Series, mean: float, std: float, config: TradingConfig) -> pd.DataFrame:
    """Generate long/short/flat positions from Vidyamurthy spread bands."""
    delta = config.vidyamurthy_threshold_sigma * std
    lower_band = mean - delta
    upper_band = mean + delta
    rows: list[dict] = []
    position = 0
    for timestamp, value in spread.astype(float).items():
        if position == 0 and value <= lower_band:
            position = 1
        elif position == 0 and value >= upper_band:
            position = -1
        elif position == 1 and value >= upper_band:
            position = 0
        elif position == -1 and value <= lower_band:
            position = 0
        rows.append(
            {
                "date": timestamp,
                "spread": float(value),
                "signal": position,
                "lower_band": lower_band,
                "upper_band": upper_band,
                "delta": delta,
            }
        )
    return pd.DataFrame(rows)
