"""Gatev-style pair trading used after ISEPT pair selection."""
import numpy as np
import pandas as pd

from common.config import TradingConfig


def fit_log_spread(prices: pd.DataFrame, asset_i: str, asset_j: str) -> tuple[float, float, float, float]:
    """Fit log(asset_i) = intercept + beta * log(asset_j) and return spread stats."""
    log_prices = np.log(prices[[asset_i, asset_j]].astype(float)).dropna()
    y = log_prices[asset_i].to_numpy(dtype=float)
    x = log_prices[asset_j].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    intercept, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    spread = build_log_spread(prices, asset_i, asset_j, float(beta), float(intercept))
    mean = float(spread.mean())
    std = float(spread.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        std = 1.0
    return float(beta), float(intercept), mean, std


def build_log_spread(prices: pd.DataFrame, asset_i: str, asset_j: str, beta: float, intercept: float) -> pd.Series:
    """Build log-spread for a pair."""
    log_prices = np.log(prices[[asset_i, asset_j]].astype(float)).dropna()
    values = log_prices[asset_i] - intercept - beta * log_prices[asset_j]
    return pd.Series(values.to_numpy(dtype=float), index=log_prices.index, name="spread")


def simulate_gatev_pair(
    history_prices: pd.DataFrame,
    trading_prices: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    config: TradingConfig,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Simulate one pair over one trading month."""
    beta, intercept, mean, std = fit_log_spread(history_prices, asset_i, asset_j)
    spread = build_log_spread(trading_prices, asset_i, asset_j, beta, intercept)
    signals = gatev_signals(spread, mean, std, config)
    returns = signal_returns(trading_prices, signals, asset_i, asset_j, beta, config)
    trades = trade_records_from_signals(signals, returns, asset_i, asset_j)
    signals["asset_i"] = asset_i
    signals["asset_j"] = asset_j
    signals["beta"] = beta
    signals["intercept"] = intercept
    signals["spread_mean"] = mean
    signals["spread_std"] = std
    signals["trade_rule"] = "gatev"
    return returns, signals, trades


def gatev_signals(spread: pd.Series, mean: float, std: float, config: TradingConfig) -> pd.DataFrame:
    """Generate Gatev-style long/short/flat positions from spread."""
    long_entry = mean - config.entry_sigma * std
    short_entry = mean + config.entry_sigma * std
    long_exit = mean - config.exit_sigma * std
    short_exit = mean + config.exit_sigma * std
    rows: list[dict] = []
    position = 0
    for timestamp, value in spread.astype(float).items():
        if position == 0 and value <= long_entry:
            position = 1
        elif position == 0 and value >= short_entry:
            position = -1
        elif position == 1 and value >= long_exit:
            position = 0
        elif position == -1 and value <= short_exit:
            position = 0
        rows.append(
            {
                "date": timestamp,
                "spread": float(value),
                "signal": position,
                "long_entry": long_entry,
                "short_entry": short_entry,
                "long_exit": long_exit,
                "short_exit": short_exit,
            }
        )
    return pd.DataFrame(rows)


def signal_returns(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    beta: float,
    config: TradingConfig,
) -> pd.Series:
    """Return daily strategy returns for one pair."""
    signal_series = signals.set_index("date")["signal"].astype(float)
    simple_returns = prices[[asset_i, asset_j]].pct_change().reindex(signal_series.index).fillna(0.0)
    gross = signal_series.shift(1).fillna(0.0) * (simple_returns[asset_i] - beta * simple_returns[asset_j])
    gross = gross.divide(1.0 + abs(beta))
    costs = signal_series.diff().abs().fillna(signal_series.abs()).multiply(config.transaction_cost_bps / 10_000.0)
    return gross.subtract(costs).rename("pair_return")


def trade_records_from_signals(signals: pd.DataFrame, returns: pd.Series, asset_i: str, asset_j: str) -> pd.DataFrame:
    """Convert signal episodes into trade records."""
    signal_series = signals.set_index("date")["signal"].astype(int)
    rows: list[dict] = []
    current = 0
    open_date = None
    episode = 0
    for timestamp, position in signal_series.items():
        if current == 0 and position != 0:
            episode += 1
            open_date = timestamp
        if current != 0 and position != current:
            rows.append(trade_record(asset_i, asset_j, episode, current, open_date, timestamp, returns))
            open_date = timestamp if position != 0 else None
            if position != 0:
                episode += 1
        current = position
    if current != 0 and open_date is not None:
        rows.append(trade_record(asset_i, asset_j, episode, current, open_date, signal_series.index[-1], returns))
    return pd.DataFrame(rows)


def trade_record(asset_i: str, asset_j: str, episode: int, position: int, open_date, close_date, returns: pd.Series) -> dict:
    """Return one trade episode row."""
    episode_returns = returns.loc[open_date:close_date]
    return {
        "asset_i": asset_i,
        "asset_j": asset_j,
        "trade_episode": episode,
        "direction": "LONG_SPREAD" if position > 0 else "SHORT_SPREAD",
        "open_date": open_date,
        "close_date": close_date,
        "holding_days": int(len(episode_returns)),
        "trade_return": float(episode_returns.sum()),
    }
