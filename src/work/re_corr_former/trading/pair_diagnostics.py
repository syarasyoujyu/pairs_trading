"""Per-pair daily diagnostics for ReCorrFormer trading runs."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from common.config import TradeConfig, WindowConfig
from trading.rule_backtest import (
    build_log_spread,
    event_identifier,
    fit_log_spread,
    formation_prices,
    future_months_window,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_pair_diagnostics(
    selected: pd.DataFrame,
    close: pd.DataFrame,
    predictions: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
    trade_signals: pd.DataFrame,
    trades: pd.DataFrame,
    windows: WindowConfig,
    trade_config: TradeConfig,
    validation_fraction: float,
    output_dir: Path,
    diagnostic_limit: int,
) -> pd.DataFrame:
    """Save per-selected-pair daily traces and diagnostic plots."""
    manifest_rows: list[dict] = []
    trace_frames: list[pd.DataFrame] = []
    trace_dir = output_dir / "pair_daily_traces"
    plot_dir = output_dir / "pair_diagnostic_plots"
    reset_diagnostic_directory(trace_dir, ".csv")
    reset_diagnostic_directory(plot_dir, ".png")
    split_ranges = model_split_ranges(predictions, candidate_metadata, validation_fraction)

    for event_number, (_, row) in enumerate(selected.iterrows(), start=1):
        if diagnostic_limit > 0 and len(manifest_rows) >= diagnostic_limit:
            break
        event_id = event_identifier(event_number, pd.Timestamp(row["date"]), str(row["asset_i"]), str(row["asset_j"]))
        event_signals = event_frame(trade_signals, event_id)
        event_trades = event_frame(trades, event_id)
        trace = build_pair_daily_trace(event_id, row, close, event_signals, event_trades, split_ranges, windows, trade_config)
        if trace.empty:
            continue
        trace_path = trace_dir / f"{event_id}.csv"
        plot_path = plot_dir / f"{event_id}.png"
        trace.to_csv(trace_path, index=False)
        save_pair_diagnostic_plot(trace, event_trades, plot_path)
        trace_frames.append(trace)
        manifest_rows.append(pair_manifest_row(trace, event_trades, trace_path, plot_path))

    combined = pd.concat(trace_frames, axis=0, ignore_index=True) if trace_frames else pd.DataFrame()
    combined.to_csv(output_dir / "pair_daily_traces.csv", index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "pair_daily_trace_manifest.csv", index=False)
    return manifest


def build_pair_daily_trace(
    event_id: str,
    row: pd.Series,
    close: pd.DataFrame,
    event_signals: pd.DataFrame,
    event_trades: pd.DataFrame,
    split_ranges: pd.DataFrame,
    windows: WindowConfig,
    trade_config: TradeConfig,
) -> pd.DataFrame:
    """Build one selected-pair daily trace from formation through trading horizon."""
    selected_date = pd.Timestamp(row["date"])
    asset_i = str(row["asset_i"])
    asset_j = str(row["asset_j"])
    if selected_date not in close.index or asset_i not in close.columns or asset_j not in close.columns:
        return pd.DataFrame()
    date_index = diagnostic_date_index(close, selected_date, split_ranges, trade_config)
    prices = close.loc[date_index, [asset_i, asset_j]].astype(float).dropna()
    if prices.empty:
        return pd.DataFrame()
    history_prices = formation_prices(row, close, trade_config)
    if history_prices.empty:
        return pd.DataFrame()
    beta, intercept, spread_mean, spread_std = fit_log_spread(history_prices, asset_i, asset_j)
    spread = build_log_spread(prices, asset_i, asset_j, beta, intercept)
    trace = base_trace_frame(event_id, row, prices, spread, beta, intercept, spread_mean, spread_std, trade_config)
    trace = add_model_split_period(trace, split_ranges)
    trace = add_relationship_indicators(trace, close, asset_i, asset_j, windows)
    trace = add_signal_and_return_columns(trace, prices, event_signals, event_trades, asset_i, asset_j, beta, trade_config)
    trace = add_threshold_columns(trace, spread_mean, spread_std, trade_config)
    return ordered_trace_columns(trace, trade_config)


def reset_diagnostic_directory(directory: Path, suffix: str) -> None:
    """Remove stale diagnostic files from a previous run."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix == suffix:
            path.unlink()


def diagnostic_date_index(
    close: pd.DataFrame,
    selected_date: pd.Timestamp,
    split_ranges: pd.DataFrame,
    config: TradeConfig,
) -> pd.Index:
    """Return dates spanning model history and forward trading horizon."""
    start_date = diagnostic_start_date(close, split_ranges)
    start_position = int(close.index.searchsorted(start_date, side="left"))
    trade_index = future_months_window(close.index, selected_date, config.trading_horizon_months)
    end_date = trade_index[-1] if len(trade_index) else selected_date
    end_position = int(close.index.get_loc(end_date))
    return close.index[start_position:end_position + 1]


def diagnostic_start_date(close: pd.DataFrame, split_ranges: pd.DataFrame) -> pd.Timestamp:
    """Return the first date to show in pair diagnostics."""
    if split_ranges.empty or "start_date" not in split_ranges.columns:
        return pd.Timestamp(close.index.min())
    split_start = pd.to_datetime(split_ranges["start_date"]).min()
    if pd.isna(split_start):
        return pd.Timestamp(close.index.min())
    return min(pd.Timestamp(close.index.min()), pd.Timestamp(split_start))


def base_trace_frame(
    event_id: str,
    row: pd.Series,
    prices: pd.DataFrame,
    spread: pd.Series,
    beta: float,
    intercept: float,
    spread_mean: float,
    spread_std: float,
    trade_config: TradeConfig,
) -> pd.DataFrame:
    """Return price, spread, and selected-pair metadata columns."""
    asset_i = str(row["asset_i"])
    asset_j = str(row["asset_j"])
    selected_date = pd.Timestamp(row["date"])
    trace = pd.DataFrame({"date": prices.index})
    trace["event_id"] = event_id
    trace["selected_date"] = selected_date
    trace["asset_i"] = asset_i
    trace["asset_j"] = asset_j
    trace["pair"] = f"{asset_i}/{asset_j}"
    trace["event_phase"] = event_phase_values(prices.index, selected_date, trade_config)
    trace["asset_i_close"] = prices[asset_i].to_numpy(dtype=float)
    trace["asset_j_close"] = prices[asset_j].to_numpy(dtype=float)
    trace["asset_i_relative_price_pct"] = relative_price_pct(prices[asset_i])
    trace["asset_j_relative_price_pct"] = relative_price_pct(prices[asset_j])
    trace["spread"] = spread.reindex(prices.index).to_numpy(dtype=float)
    trace["spread_mean"] = spread_mean
    trace["spread_std"] = spread_std
    trace["spread_z"] = stable_zscore(trace["spread"], spread_mean, spread_std)
    trace["gamma"] = beta
    trace["beta"] = beta
    trace["intercept"] = intercept
    trace["trade_rule"] = trade_config.trade_rule
    trace["selected_rho_long"] = row_float(row, "rho_long", np.nan)
    trace["selected_rho_now"] = row_float(row, "rho_now", np.nan)
    trace["selected_gap"] = row_float(row, "gap", np.nan)
    trace["selected_spread_z"] = row_float(row, "spread_z", np.nan)
    trace["selected_final_score"] = row_float(row, "final_score", np.nan)
    trace["selected_pred_corr"] = row_float(row, "pred_corr", np.nan)
    trace["selected_predicted_sharpe"] = row_float(row, "predicted_sharpe", np.nan)
    return trace


def event_phase_values(index: pd.Index, selected_date: pd.Timestamp, config: TradeConfig) -> list[str]:
    """Return formation, selection, and trading phase labels."""
    trade_index = future_months_window(index, selected_date, config.trading_horizon_months)
    trade_dates = set(pd.to_datetime(trade_index))
    phases = []
    for date in pd.to_datetime(index):
        if date < selected_date:
            phases.append("formation")
        elif date == selected_date:
            phases.append("selection_date")
        elif date in trade_dates:
            phases.append("trading")
        else:
            phases.append("after_trading")
    return phases


def relative_price_pct(price: pd.Series) -> np.ndarray:
    """Return percent movement from the first visible date."""
    base = float(price.iloc[0])
    if base == 0.0 or not np.isfinite(base):
        base = 1.0
    return price.divide(base).subtract(1.0).multiply(100.0).to_numpy(dtype=float)


def stable_zscore(values: pd.Series, mean: float, std: float) -> np.ndarray:
    """Return z-scores using stable spread statistics."""
    scale = std if std != 0.0 and np.isfinite(std) else 1.0
    return values.astype(float).subtract(mean).divide(scale).to_numpy(dtype=float)


def model_split_ranges(
    predictions: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
    validation_fraction: float,
) -> pd.DataFrame:
    """Return chronological split ranges from prediction metadata."""
    if prediction_splits_include_training_periods(predictions):
        return split_ranges_from_frame(predictions)
    if not candidate_metadata.empty and "date" in candidate_metadata.columns:
        return chronological_candidate_split_ranges(candidate_metadata, validation_fraction)
    return pd.DataFrame(columns=["split", "start_date", "end_date"])


def prediction_splits_include_training_periods(predictions: pd.DataFrame) -> bool:
    """Return whether predictions carry train/validation/test split labels."""
    if predictions.empty or "split" not in predictions.columns or "date" not in predictions.columns:
        return False
    splits = set(predictions["split"].dropna().astype(str))
    return "train" in splits and ("validation" in splits or "val" in splits) and "test" in splits


def split_ranges_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return split date ranges from a frame with date and split columns."""
    frame = frame.loc[:, ["date", "split"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    ranges = (
        frame.groupby("split", as_index=False)
        .agg(start_date=("date", "min"), end_date=("date", "max"))
        .sort_values("start_date")
        .reset_index(drop=True)
    )
    return continuous_split_ranges(ranges)


def chronological_candidate_split_ranges(candidate_metadata: pd.DataFrame, validation_fraction: float) -> pd.DataFrame:
    """Return train/validation/test ranges from candidate dates."""
    dates = pd.Index(sorted(pd.to_datetime(candidate_metadata["date"]).dropna().unique()))
    if dates.empty:
        return pd.DataFrame(columns=["split", "start_date", "end_date"])
    train_count = max(1, int(len(dates) * (1.0 - 2.0 * validation_fraction)))
    validation_count = max(1, int(len(dates) * validation_fraction))
    train_dates = dates[:train_count]
    validation_dates = dates[train_count:train_count + validation_count]
    test_dates = dates[train_count + validation_count:]
    rows = split_range_rows(train_dates, validation_dates, test_dates)
    return continuous_split_ranges(pd.DataFrame(rows))


def split_range_rows(train_dates: pd.Index, validation_dates: pd.Index, test_dates: pd.Index) -> list[dict]:
    """Return non-empty split range rows."""
    rows = []
    for split_name, dates in (("train", train_dates), ("validation", validation_dates), ("test", test_dates)):
        if len(dates):
            rows.append({"split": split_name, "start_date": dates.min(), "end_date": dates.max()})
    return rows


def continuous_split_ranges(ranges: pd.DataFrame) -> pd.DataFrame:
    """Extend split ranges so daily traces do not have gaps between sampled dates."""
    if ranges.empty:
        return ranges
    frame = ranges.sort_values("start_date").reset_index(drop=True).copy()
    frame["start_date"] = pd.to_datetime(frame["start_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    for index in range(len(frame) - 1):
        next_start = pd.Timestamp(frame.loc[index + 1, "start_date"])
        frame.loc[index, "end_date"] = next_start - pd.Timedelta(days=1)
    return frame


def add_model_split_period(trace: pd.DataFrame, split_ranges: pd.DataFrame) -> pd.DataFrame:
    """Add a model split period label for every visible date."""
    frame = trace.copy()
    frame["model_split_period"] = [split_for_date(pd.Timestamp(date), split_ranges) for date in frame["date"]]
    return frame


def split_for_date(date: pd.Timestamp, split_ranges: pd.DataFrame) -> str:
    """Return split name covering a date."""
    if split_ranges.empty:
        return "unknown"
    for _, row in split_ranges.iterrows():
        if pd.Timestamp(row["start_date"]) <= date <= pd.Timestamp(row["end_date"]):
            return str(row["split"])
    first = split_ranges.iloc[0]
    last = split_ranges.iloc[-1]
    if date < pd.Timestamp(first["start_date"]):
        return f"before_{first['split']}"
    if date > pd.Timestamp(last["end_date"]):
        return f"after_{last['split']}"
    return "between_splits"


def add_relationship_indicators(
    trace: pd.DataFrame,
    close: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    windows: WindowConfig,
) -> pd.DataFrame:
    """Add rolling rho_long, rho_now, and gap values for each date."""
    indicators = rolling_relationship_indicators(close, asset_i, asset_j, pd.to_datetime(trace["date"]), windows)
    frame = trace.merge(indicators, on="date", how="left")
    return frame


def rolling_relationship_indicators(
    close: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    dates: pd.Series,
    windows: WindowConfig,
) -> pd.DataFrame:
    """Compute pair-level rolling long/short correlation indicators."""
    returns = close[[asset_i, asset_j]].astype(float).pct_change().dropna()
    rows = []
    for date in pd.to_datetime(dates):
        rows.append(relationship_indicator_row(returns, asset_i, asset_j, date, windows))
    return pd.DataFrame(rows)


def relationship_indicator_row(
    returns: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    date: pd.Timestamp,
    windows: WindowConfig,
) -> dict:
    """Return one date's rho_long, rho_now, and gap."""
    row = {"date": date}
    if date not in returns.index:
        row.update(empty_relationship_values())
        return row
    position = int(returns.index.get_loc(date))
    rho_long = lagged_pair_corr(returns, asset_i, asset_j, position, windows.long_corr_min_lag, windows.long_corr_window)
    rho_now = lagged_pair_corr(returns, asset_i, asset_j, position, windows.short_corr_min_lag, windows.short_corr_window)
    row["rho_long"] = rho_long
    row["rho_now"] = rho_now
    row["gap"] = rho_long - rho_now if np.isfinite(rho_long) and np.isfinite(rho_now) else np.nan
    return row


def lagged_pair_corr(
    returns: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    position: int,
    min_lag: int,
    max_lag: int,
) -> float:
    """Return lagged correlation for one pair at one decision position."""
    start = position - max_lag
    end = position - min_lag + 1
    if start < 0 or end <= start:
        return np.nan
    window = returns[[asset_i, asset_j]].iloc[start:end].dropna()
    if len(window) < 2:
        return np.nan
    return float(window[asset_i].corr(window[asset_j]))


def empty_relationship_values() -> dict:
    """Return missing relationship indicator values."""
    return {"rho_long": np.nan, "rho_now": np.nan, "gap": np.nan}


def add_signal_and_return_columns(
    trace: pd.DataFrame,
    prices: pd.DataFrame,
    event_signals: pd.DataFrame,
    event_trades: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    beta: float,
    trade_config: TradeConfig,
) -> pd.DataFrame:
    """Add signal, trade action, raw pair return, and strategy return columns."""
    frame = trace.copy()
    signal_series = event_signal_series(frame["date"], event_signals)
    raw_return = raw_pair_return(prices, asset_i, asset_j, beta).reindex(pd.to_datetime(frame["date"])).fillna(0.0)
    costs = signal_series.diff().abs().fillna(signal_series.abs()).multiply(trade_config.transaction_cost_bps / 10_000.0)
    strategy_return = signal_series.shift(1).fillna(0.0).multiply(raw_return).subtract(costs)
    frame["signal"] = signal_series.to_numpy(dtype=int)
    frame["position_name"] = [position_name(value) for value in frame["signal"]]
    frame["trade_action"] = trade_action_values(signal_series)
    frame["trade_marker"] = trade_marker_values(frame["date"], event_trades)
    frame["raw_pair_daily_return"] = raw_return.to_numpy(dtype=float)
    frame["raw_pair_cumulative_return"] = raw_return.add(1.0).cumprod().subtract(1.0).to_numpy(dtype=float)
    frame["strategy_daily_return"] = strategy_return.to_numpy(dtype=float)
    frame["strategy_cumulative_return"] = strategy_return.add(1.0).cumprod().subtract(1.0).to_numpy(dtype=float)
    frame["transaction_cost"] = costs.to_numpy(dtype=float)
    frame["is_trade_date"] = frame["trade_marker"].astype(str).ne("")
    return frame


def event_signal_series(dates: pd.Series, event_signals: pd.DataFrame) -> pd.Series:
    """Return signal values indexed by trace dates."""
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if event_signals.empty:
        return pd.Series(0, index=index, dtype=int)
    frame = event_signals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")["signal"].astype(int).reindex(index).fillna(0).astype(int)


def raw_pair_return(prices: pd.DataFrame, asset_i: str, asset_j: str, beta: float) -> pd.Series:
    """Return unpositioned hedge-adjusted pair daily return."""
    simple_returns = prices[[asset_i, asset_j]].pct_change().fillna(0.0)
    values = simple_returns[asset_i].subtract(beta * simple_returns[asset_j]).divide(1.0 + abs(beta))
    return values.rename("raw_pair_daily_return")


def position_name(signal: int) -> str:
    """Return a readable position name for a signal."""
    if signal > 0:
        return "LONG_SPREAD"
    if signal < 0:
        return "SHORT_SPREAD"
    return "FLAT"


def trade_action_values(signal_series: pd.Series) -> list[str]:
    """Return open, close, and switch action labels from signal changes."""
    actions = []
    previous = 0
    for current in signal_series.astype(int):
        actions.append(trade_action(previous, int(current)))
        previous = int(current)
    return actions


def trade_action(previous: int, current: int) -> str:
    """Return one action label from previous and current signal."""
    if previous == current:
        return ""
    if previous == 0 and current > 0:
        return "open_long_spread"
    if previous == 0 and current < 0:
        return "open_short_spread"
    if previous > 0 and current == 0:
        return "close_long_spread"
    if previous < 0 and current == 0:
        return "close_short_spread"
    if previous > 0 and current < 0:
        return "switch_long_to_short"
    if previous < 0 and current > 0:
        return "switch_short_to_long"
    return "signal_change"


def trade_marker_values(dates: pd.Series, event_trades: pd.DataFrame) -> list[str]:
    """Return trade episode open and close markers for trace dates."""
    marker_by_date: dict[pd.Timestamp, list[str]] = {}
    if not event_trades.empty:
        for _, trade in event_trades.iterrows():
            add_marker(marker_by_date, pd.Timestamp(trade["open_date"]), f"open_{trade['direction']}")
            add_marker(marker_by_date, pd.Timestamp(trade["close_date"]), f"close_{trade['direction']}")
    return [";".join(marker_by_date.get(pd.Timestamp(date), [])) for date in pd.to_datetime(dates)]


def add_marker(markers: dict[pd.Timestamp, list[str]], date: pd.Timestamp, value: str) -> None:
    """Add a trade marker value on a date."""
    markers.setdefault(date, []).append(value)


def add_threshold_columns(trace: pd.DataFrame, spread_mean: float, spread_std: float, config: TradeConfig) -> pd.DataFrame:
    """Add trade-rule threshold columns across the full trace."""
    frame = trace.copy()
    if config.trade_rule == "vidyamurthy":
        delta = config.vidyamurthy_threshold_sigma * spread_std
        frame["lower_band"] = spread_mean - delta
        frame["upper_band"] = spread_mean + delta
        frame["delta"] = delta
    elif config.trade_rule == "gatev":
        frame["long_entry"] = spread_mean - config.entry_sigma * spread_std
        frame["short_entry"] = spread_mean + config.entry_sigma * spread_std
        frame["long_exit"] = spread_mean - config.exit_sigma * spread_std
        frame["short_exit"] = spread_mean + config.exit_sigma * spread_std
    return frame


def ordered_trace_columns(trace: pd.DataFrame, config: TradeConfig) -> pd.DataFrame:
    """Return trace columns in a stable order."""
    base_columns = [
        "event_id",
        "selected_date",
        "date",
        "model_split_period",
        "event_phase",
        "asset_i",
        "asset_j",
        "pair",
        "trade_rule",
        "signal",
        "position_name",
        "trade_action",
        "trade_marker",
        "raw_pair_daily_return",
        "strategy_daily_return",
        "strategy_cumulative_return",
        "transaction_cost",
        "is_trade_date",
        "spread",
        "spread_z",
        "spread_mean",
        "spread_std",
        "rho_long",
        "rho_now",
        "gap",
        "selected_rho_long",
        "selected_rho_now",
        "selected_gap",
        "selected_spread_z",
        "selected_final_score",
        "selected_pred_corr",
        "selected_predicted_sharpe",
        "gamma",
        "beta",
        "intercept",
    ]
    price_columns = ["asset_i_close", "asset_j_close", "asset_i_relative_price_pct", "asset_j_relative_price_pct"]
    threshold_columns = rule_threshold_columns(config)
    remaining = [column for column in trace.columns if column not in base_columns + price_columns + threshold_columns]
    return trace.loc[:, base_columns + price_columns + threshold_columns + remaining]


def rule_threshold_columns(config: TradeConfig) -> list[str]:
    """Return threshold columns for a trade rule."""
    if config.trade_rule == "vidyamurthy":
        return ["lower_band", "upper_band", "delta"]
    if config.trade_rule == "gatev":
        return ["long_entry", "short_entry", "long_exit", "short_exit"]
    return []


def pair_manifest_row(trace: pd.DataFrame, event_trades: pd.DataFrame, trace_path: Path, plot_path: Path) -> dict:
    """Return manifest metadata for one pair trace."""
    first = trace.iloc[0]
    return {
        "event_id": first["event_id"],
        "selected_date": first["selected_date"],
        "asset_i": first["asset_i"],
        "asset_j": first["asset_j"],
        "pair": first["pair"],
        "trace_start_date": trace["date"].min(),
        "trace_end_date": trace["date"].max(),
        "trade_count": int(len(event_trades)),
        "total_strategy_return_pct": float(trace["strategy_cumulative_return"].iloc[-1] * 100.0),
        "trace_path": str(trace_path),
        "plot_path": str(plot_path),
    }


def save_pair_diagnostic_plot(trace: pd.DataFrame, event_trades: pd.DataFrame, image_path: Path) -> None:
    """Save a selected-pair diagnostic plot."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    frame = trace.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    plot_relative_price_panel(axes[0], frame, event_trades)
    plot_relationship_panel(axes[1], frame)
    plot_spread_signal_panel(axes[2], frame)
    plot_return_panel(axes[3], frame)
    fig.suptitle(f"{frame['event_id'].iloc[0]} | {frame['pair'].iloc[0]}", fontsize=13)
    fig.savefig(image_path, dpi=150)
    plt.close(fig)


def plot_relative_price_panel(axis, trace: pd.DataFrame, event_trades: pd.DataFrame) -> None:
    """Plot relative price movement and trade dates."""
    apply_split_background(axis, trace)
    asset_i = str(trace["asset_i"].iloc[0])
    asset_j = str(trace["asset_j"].iloc[0])
    axis.plot(trace["date"], trace["asset_i_relative_price_pct"], label=f"{asset_i} relative %", linewidth=1.4)
    axis.plot(trace["date"], trace["asset_j_relative_price_pct"], label=f"{asset_j} relative %", linewidth=1.4)
    mark_selected_date(axis, trace)
    mark_trade_dates(axis, event_trades)
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    axis.set_ylabel("Relative price (%)")
    axis.legend(loc="upper left", ncols=2)
    axis.grid(alpha=0.25)


def plot_relationship_panel(axis, trace: pd.DataFrame) -> None:
    """Plot rolling relationship-gap indicators."""
    apply_split_background(axis, trace)
    axis.plot(trace["date"], trace["rho_long"], label="rho_long", linewidth=1.3)
    axis.plot(trace["date"], trace["rho_now"], label="rho_now", linewidth=1.3)
    axis.plot(trace["date"], trace["gap"], label="gap", linewidth=1.3)
    mark_selected_date(axis, trace)
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    axis.set_ylabel("Correlation")
    axis.legend(loc="upper left", ncols=3)
    axis.grid(alpha=0.25)


def plot_spread_signal_panel(axis, trace: pd.DataFrame) -> None:
    """Plot spread, trade thresholds, and position changes."""
    apply_split_background(axis, trace)
    axis.plot(trace["date"], trace["spread"], label="spread", color="#1f2937", linewidth=1.2)
    axis.plot(trace["date"], trace["spread_mean"], label="spread_mean", color="#6b7280", linewidth=0.9)
    for column in threshold_plot_columns(trace):
        axis.plot(trace["date"], trace[column], label=column, linewidth=0.9, alpha=0.85)
    plot_signal_markers(axis, trace)
    mark_selected_date(axis, trace)
    axis.set_ylabel("Spread")
    axis.legend(loc="upper left", ncols=3)
    axis.grid(alpha=0.25)


def plot_return_panel(axis, trace: pd.DataFrame) -> None:
    """Plot strategy daily and cumulative return."""
    apply_split_background(axis, trace)
    axis.bar(trace["date"], trace["strategy_daily_return"].multiply(100.0), label="strategy daily return %", color="#94a3b8", width=1.0)
    twin = axis.twinx()
    plot_cumulative_return_line(
        twin,
        trace,
        "strategy_cumulative_return",
        "strategy cumulative return %",
        "#0f766e",
        1.5,
    )
    mark_selected_date(axis, trace)
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    axis.set_ylabel("Daily return (%)")
    twin.set_ylabel("Cumulative return (%)")
    axis.grid(alpha=0.25)
    merge_legends(axis, twin)


def plot_cumulative_return_line(axis, trace: pd.DataFrame, column: str, label: str, color: str, linewidth: float) -> None:
    """Plot a cumulative return line with an explicit 0% baseline point."""
    dates, values = cumulative_return_plot_series(trace, column)
    axis.plot(dates, values, label=label, color=color, linewidth=linewidth)


def cumulative_return_plot_series(trace: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Return cumulative return plot values with a synthetic zero baseline."""
    dates = pd.Series(pd.to_datetime(trace["date"])).reset_index(drop=True)
    values = trace[column].astype(float).multiply(100.0).reset_index(drop=True)
    if dates.empty:
        return dates, values
    baseline_date = cumulative_baseline_date(dates)
    plot_dates = pd.concat([pd.Series([baseline_date]), dates], ignore_index=True)
    plot_values = pd.concat([pd.Series([0.0]), values], ignore_index=True)
    return plot_dates, plot_values


def cumulative_baseline_date(dates: pd.Series) -> pd.Timestamp:
    """Return the timestamp used for the left-edge 0% cumulative baseline."""
    if len(dates) > 1:
        first_gap = dates.iloc[1] - dates.iloc[0]
        if first_gap > pd.Timedelta(0):
            return pd.Timestamp(dates.iloc[0] - first_gap)
    return pd.Timestamp(dates.iloc[0] - pd.Timedelta(days=1))


def threshold_plot_columns(trace: pd.DataFrame) -> list[str]:
    """Return threshold columns present in a trace."""
    candidates = ["lower_band", "upper_band", "long_entry", "short_entry", "long_exit", "short_exit"]
    return [column for column in candidates if column in trace.columns]


def plot_signal_markers(axis, trace: pd.DataFrame) -> None:
    """Plot long and short spread signal markers."""
    long_rows = trace[trace["trade_action"].str.contains("long", na=False)]
    short_rows = trace[trace["trade_action"].str.contains("short", na=False)]
    if not long_rows.empty:
        axis.scatter(long_rows["date"], long_rows["spread"], marker="^", color="#16a34a", s=28, label="long action", zorder=5)
    if not short_rows.empty:
        axis.scatter(short_rows["date"], short_rows["spread"], marker="v", color="#dc2626", s=28, label="short action", zorder=5)


def apply_split_background(axis, trace: pd.DataFrame) -> None:
    """Shade model split periods behind a plot."""
    for split_name, group in trace.groupby("model_split_period", sort=False):
        color = split_color(str(split_name))
        axis.axvspan(pd.to_datetime(group["date"]).min(), pd.to_datetime(group["date"]).max(), color=color, alpha=0.10, linewidth=0)


def split_color(split_name: str) -> str:
    """Return a background color for a split label."""
    if "train" in split_name:
        return "#16a34a"
    if "validation" in split_name or "val" in split_name:
        return "#ca8a04"
    if "test" in split_name:
        return "#2563eb"
    return "#64748b"


def mark_selected_date(axis, trace: pd.DataFrame) -> None:
    """Mark selected date on an axis."""
    selected_date = pd.Timestamp(trace["selected_date"].iloc[0])
    axis.axvline(selected_date, color="black", linewidth=1.0, linestyle="--", alpha=0.75)


def mark_trade_dates(axis, trades: pd.DataFrame) -> None:
    """Mark trade open and close dates on an axis."""
    if trades.empty:
        return
    for _, trade in trades.iterrows():
        axis.axvline(pd.Timestamp(trade["open_date"]), color="#16a34a", linewidth=0.8, alpha=0.35)
        axis.axvline(pd.Timestamp(trade["close_date"]), color="#dc2626", linewidth=0.8, alpha=0.35)


def merge_legends(axis, twin_axis) -> None:
    """Merge legends from two y-axes."""
    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin_axis.get_legend_handles_labels()
    axis.legend(handles + twin_handles, labels + twin_labels, loc="upper left")


def event_frame(frame: pd.DataFrame, event_id: str) -> pd.DataFrame:
    """Return rows for one event id."""
    if frame.empty or "event_id" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["event_id"] == event_id].copy()


def row_float(row: pd.Series, column: str, default: float) -> float:
    """Return a numeric row value or default."""
    if column not in row.index or pd.isna(row[column]):
        return float(default)
    return float(row[column])
