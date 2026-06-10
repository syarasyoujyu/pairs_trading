"""ISEPT-style execution rules for ReCorrFormer-selected pairs."""
import numpy as np
import pandas as pd

from common.config import TradeConfig


def backtest_rule_pairs(
    selected: pd.DataFrame,
    close: pd.DataFrame,
    trade_config: TradeConfig,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Backtest selected pairs with the configured ISEPT-style execution rule."""
    event_returns: list[pd.Series] = []
    trade_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []

    for event_number, (_, row) in enumerate(selected.iterrows(), start=1):
        result = run_trade_event(event_number, row, close, trade_config)
        if result["returns"].empty:
            continue
        event_returns.append(result["returns"])
        trade_frames.append(result["trades"])
        signal_frames.append(result["signals"])

    portfolio = portfolio_from_event_returns(event_returns, trade_config.trade_rule)
    trades = concat_or_empty(trade_frames)
    signals = concat_or_empty(signal_frames)
    return portfolio, trades, signals


def run_trade_event(event_number: int, row: pd.Series, close: pd.DataFrame, config: TradeConfig) -> dict:
    """Run one ReCorrFormer-selected pair through the configured execution rule."""
    asset_i = str(row["asset_i"])
    asset_j = str(row["asset_j"])
    selected_date = pd.Timestamp(row["date"])
    event_id = event_identifier(event_number, selected_date, asset_i, asset_j)
    history_prices = formation_prices(row, close, config)
    trading_prices = trading_window_prices(row, close, config)
    if history_prices.empty or trading_prices.empty:
        return empty_event_result(event_id)
    beta, intercept, mean, std = fit_log_spread(history_prices, asset_i, asset_j)
    trading_spread = build_log_spread(trading_prices, asset_i, asset_j, beta, intercept)
    signals = execution_rule_signals(trading_spread, mean, std, config)
    signals = enrich_signal_frame(signals, event_id, row, beta, intercept, mean, std, config.trade_rule)
    returns = event_returns_from_signals(trading_prices, signals, asset_i, asset_j, beta, config)
    trades = event_trades_from_signals(signals, returns)
    return {"returns": returns.rename(event_id), "trades": trades, "signals": signals}


def formation_prices(row: pd.Series, close: pd.DataFrame, config: TradeConfig) -> pd.DataFrame:
    """Return the formation window ending at the selected date."""
    selected_date = pd.Timestamp(row["date"])
    decision_position = int(close.index.get_loc(selected_date))
    start = max(0, decision_position - config.formation_window + 1)
    assets = [row["asset_i"], row["asset_j"]]
    return close[assets].iloc[start:decision_position + 1].astype(float).dropna()


def trading_window_prices(row: pd.Series, close: pd.DataFrame, config: TradeConfig) -> pd.DataFrame:
    """Return post-selection trading prices over the configured calendar horizon."""
    selected_date = pd.Timestamp(row["date"])
    trade_index = future_months_window(close.index, selected_date, config.trading_horizon_months)
    assets = [row["asset_i"], row["asset_j"]]
    return close.loc[trade_index, assets].astype(float).dropna()


def future_months_window(index: pd.Index, selected_date: pd.Timestamp, horizon_months: int) -> pd.Index:
    """Return ISEPT-style trading dates after selection through the horizon months."""
    dates = pd.Index(pd.to_datetime(index)).sort_values()
    decision_position = int(dates.get_loc(selected_date))
    start_position = decision_position + 1
    if start_position >= len(dates):
        return pd.Index([])
    start_date = dates[start_position]
    start_period = start_date.to_period("M")
    end_period = start_period + max(1, horizon_months) - 1
    date_series = pd.Series(dates, index=dates)
    periods = date_series.dt.to_period("M")
    mask = (date_series >= start_date) & (periods >= start_period) & (periods <= end_period)
    return dates[mask.to_numpy()]


def fit_log_spread(prices: pd.DataFrame, asset_i: str, asset_j: str) -> tuple[float, float, float, float]:
    """Fit log(asset_i) = intercept + beta * log(asset_j) and return spread stats."""
    log_prices = np.log(prices[[asset_i, asset_j]].astype(float)).dropna()
    if len(log_prices) < 3:
        return 0.0, 0.0, 0.0, 1.0
    y = log_prices[asset_i].to_numpy(dtype=float)
    x = log_prices[asset_j].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    intercept, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    spread = build_log_spread(prices, asset_i, asset_j, float(beta), float(intercept))
    mean, std = spread_mean_std(spread)
    return float(beta), float(intercept), mean, std


def build_log_spread(prices: pd.DataFrame, asset_i: str, asset_j: str, beta: float, intercept: float) -> pd.Series:
    """Build log(asset_i) - intercept - beta * log(asset_j)."""
    log_prices = np.log(prices[[asset_i, asset_j]].astype(float)).dropna()
    values = log_prices[asset_i] - intercept - beta * log_prices[asset_j]
    return pd.Series(values.to_numpy(dtype=float), index=log_prices.index, name="spread")


def spread_mean_std(spread: pd.Series) -> tuple[float, float]:
    """Return stable spread mean and standard deviation."""
    mean = float(spread.mean())
    std = float(spread.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        std = 1.0
    return mean, std


def execution_rule_signals(spread: pd.Series, mean: float, std: float, config: TradeConfig) -> pd.DataFrame:
    """Dispatch to the configured signal rule."""
    if config.trade_rule == "vidyamurthy":
        return vidyamurthy_signals(spread, mean, std, config)
    if config.trade_rule == "gatev":
        return gatev_signals(spread, mean, std, config)
    raise ValueError(f"Unsupported trade_rule: {config.trade_rule}")


def vidyamurthy_signals(spread: pd.Series, mean: float, std: float, config: TradeConfig) -> pd.DataFrame:
    """Generate Vidyamurthy long/short/flat signals."""
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


def gatev_signals(spread: pd.Series, mean: float, std: float, config: TradeConfig) -> pd.DataFrame:
    """Generate Gatev-style long/short/flat signals from spread."""
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


def event_returns_from_signals(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    beta: float,
    config: TradeConfig,
) -> pd.Series:
    """Return net daily pair returns from signals."""
    if signals.empty:
        return pd.Series(dtype=float, name="pair_return")
    signal_series = signals.set_index("date")["signal"].astype(float)
    simple_returns = prices[[asset_i, asset_j]].pct_change().reindex(signal_series.index).fillna(0.0)
    gross = signal_series.shift(1).fillna(0.0) * (simple_returns[asset_i] - beta * simple_returns[asset_j])
    gross = gross.divide(1.0 + abs(beta))
    costs = signal_series.diff().abs().fillna(signal_series.abs()).multiply(config.transaction_cost_bps / 10_000.0)
    return gross.subtract(costs).rename("pair_return")


def event_trades_from_signals(signals: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """Summarize non-flat holding episodes."""
    signal_series = signals.set_index("date")["signal"].astype(int)
    rows: list[dict] = []
    episode_id = 0
    current_position = 0
    open_date = None
    for timestamp, position in signal_series.items():
        if current_position == 0 and position != 0:
            episode_id += 1
            open_date = timestamp
        if current_position != 0 and position != current_position:
            rows.append(trade_episode_record(signals, returns, episode_id, current_position, open_date, timestamp))
            open_date = timestamp if position != 0 else None
            if position != 0:
                episode_id += 1
        current_position = position
    if current_position != 0 and open_date is not None:
        rows.append(trade_episode_record(signals, returns, episode_id, current_position, open_date, signal_series.index[-1]))
    return pd.DataFrame(rows)


def trade_episode_record(signals: pd.DataFrame, returns: pd.Series, episode_id: int, position: int, open_date, close_date) -> dict:
    """Return one trade episode record."""
    first = signals.iloc[0]
    episode_returns = returns.loc[open_date:close_date]
    return {
        "event_id": first["event_id"],
        "trade_episode": episode_id,
        "selected_date": first["selected_date"],
        "asset_i": first["asset_i"],
        "asset_j": first["asset_j"],
        "direction": "LONG_SPREAD" if position > 0 else "SHORT_SPREAD",
        "open_date": open_date,
        "close_date": close_date,
        "holding_days": int(len(episode_returns)),
        "trade_return": float(episode_returns.sum()),
        "pair_selection_score": first["pair_selection_score"],
        "pred_corr": first["pred_corr"],
        "predicted_sharpe": first["predicted_sharpe"],
        "trade_rule": first["trade_rule"],
    }


def enrich_signal_frame(
    signals: pd.DataFrame,
    event_id: str,
    row: pd.Series,
    beta: float,
    intercept: float,
    spread_mean: float,
    spread_std: float,
    trade_rule: str,
) -> pd.DataFrame:
    """Add selected-pair metadata to an execution signal frame."""
    frame = signals.copy()
    frame.insert(0, "event_id", event_id)
    frame.insert(1, "selected_date", pd.Timestamp(row["date"]))
    frame.insert(2, "asset_i", row["asset_i"])
    frame.insert(3, "asset_j", row["asset_j"])
    frame["pair"] = f"{row['asset_i']}/{row['asset_j']}"
    frame["pair_selection_score"] = row_float(row, "final_score", 0.0)
    frame["pred_corr"] = row_float(row, "pred_corr", np.nan)
    frame["predicted_sharpe"] = row_float(row, "predicted_sharpe", np.nan)
    frame["gamma"] = beta
    frame["beta"] = beta
    frame["intercept"] = intercept
    frame["spread_mean"] = spread_mean
    frame["spread_std"] = spread_std
    frame["trade_rule"] = trade_rule
    return frame


def portfolio_from_event_returns(event_returns: list[pd.Series], trade_rule: str) -> pd.Series:
    """Average event returns into a portfolio daily return series."""
    name = f"re_corr_former_{trade_rule}_return"
    if not event_returns:
        return pd.Series(dtype=float, name=name)
    frame = pd.concat(event_returns, axis=1).fillna(0.0)
    return frame.mean(axis=1).rename(name)


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames or return an empty frame."""
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def empty_event_result(event_id: str) -> dict:
    """Return an empty event result."""
    return {"returns": pd.Series(dtype=float, name=event_id), "trades": pd.DataFrame(), "signals": pd.DataFrame()}


def row_float(row: pd.Series, column: str, default: float) -> float:
    """Return a finite float from a row when the column exists."""
    if column not in row.index:
        return float(default)
    value = row[column]
    if pd.isna(value):
        return float(default)
    return float(value)


def event_identifier(event_number: int, selected_date: pd.Timestamp, asset_i: str, asset_j: str) -> str:
    """Return a stable selected-pair event identifier."""
    return f"event_{event_number:04d}_{selected_date.strftime('%Y%m%d')}_{asset_i}_{asset_j}"
