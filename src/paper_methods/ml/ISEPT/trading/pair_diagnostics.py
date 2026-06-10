"""Per-pair daily diagnostics for ISEPT trading runs."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from common.config import ImageConfig, TradingConfig
from images.calendar import future_months_window
from labels.pair_labels import history_window, pair_price_frame
from trading.gatev import build_log_spread, fit_log_spread
from trading.simulation import event_identifier


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_pair_diagnostics(
    selected_pairs: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    image_config: ImageConfig,
    trading_config: TradingConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Save per-selected-event daily traces and diagnostic plots."""
    trace_dir = output_dir / "pair_daily_traces"
    plot_dir = output_dir / "pair_diagnostic_plots"
    reset_directory(trace_dir, ".csv")
    reset_directory(plot_dir, ".png")
    trace_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    index = next(iter(panel.values())).index

    for event_number, (_, row) in enumerate(selected_pairs.iterrows(), start=1):
        month_end = pd.Timestamp(row["month_end"])
        asset_i = str(row["asset_i"])
        asset_j = str(row["asset_j"])
        event_id = event_identifier(event_number, month_end, asset_i, asset_j)
        event_signals = event_frame(signals, event_id)
        event_trades = event_frame(trades, event_id)
        trace = build_pair_daily_trace(
            event_id,
            row,
            panel,
            index,
            event_signals,
            event_trades,
            image_config,
            trading_config,
        )
        if trace.empty:
            continue
        trace_path = trace_dir / f"{event_id}.csv"
        plot_path = plot_dir / f"{event_id}.png"
        trace.to_csv(trace_path, index=False)
        save_pair_diagnostic_plot(trace, event_trades, plot_path)
        trace_frames.append(trace)
        manifest_rows.append(manifest_row(trace, event_trades, trace_path, plot_path))

    combined = pd.concat(trace_frames, axis=0, ignore_index=True) if trace_frames else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    combined.to_csv(output_dir / "pair_daily_traces.csv", index=False)
    manifest.to_csv(output_dir / "pair_daily_trace_manifest.csv", index=False)
    return manifest


def build_pair_daily_trace(
    event_id: str,
    selected_row: pd.Series,
    panel: dict[str, pd.DataFrame],
    index: pd.Index,
    event_signals: pd.DataFrame,
    event_trades: pd.DataFrame,
    image_config: ImageConfig,
    trading_config: TradingConfig,
) -> pd.DataFrame:
    """Build one selected event's daily diagnostic trace."""
    month_end = pd.Timestamp(selected_row["month_end"])
    asset_i = str(selected_row["asset_i"])
    asset_j = str(selected_row["asset_j"])
    history_index = history_window(index, month_end, image_config.lookback_bars)
    trade_index = future_months_window(index, month_end, trading_config.trading_horizon_months)
    visible_index = diagnostic_visible_index(index, month_end, trade_index)
    split_history_index = index[pd.to_datetime(index) <= month_end]
    history_prices = pair_price_frame(panel, asset_i, asset_j, history_index)
    visible_prices = pair_price_frame(panel, asset_i, asset_j, visible_index)
    if history_prices.empty or visible_prices.empty:
        return pd.DataFrame()
    beta, intercept, spread_mean, spread_std = fit_log_spread(history_prices, asset_i, asset_j)
    spread = build_log_spread(visible_prices, asset_i, asset_j, beta, intercept)
    trace = base_trace_frame(
        event_id,
        selected_row,
        visible_prices,
        spread,
        beta,
        intercept,
        spread_mean,
        spread_std,
        split_history_index,
        trade_index,
        trading_config,
    )
    trace = add_signal_return_columns(trace, visible_prices, event_signals, event_trades, asset_i, asset_j, beta, trading_config)
    trace = add_threshold_columns(trace, event_signals, trading_config)
    return ordered_trace_columns(trace, trading_config)


def diagnostic_visible_index(index: pd.Index, month_end: pd.Timestamp, trade_index: pd.Index) -> pd.Index:
    """Return the full visible history through the trading horizon."""
    available = pd.DatetimeIndex(pd.to_datetime(index))
    end_date = pd.Timestamp(trade_index[-1]) if len(trade_index) else pd.Timestamp(month_end)
    return pd.Index(available[available <= end_date]).sort_values()


def base_trace_frame(
    event_id: str,
    selected_row: pd.Series,
    prices: pd.DataFrame,
    spread: pd.Series,
    beta: float,
    intercept: float,
    spread_mean: float,
    spread_std: float,
    history_index: pd.Index,
    trade_index: pd.Index,
    trading_config: TradingConfig,
) -> pd.DataFrame:
    """Return base metadata, price, and spread columns."""
    asset_i = str(selected_row["asset_i"])
    asset_j = str(selected_row["asset_j"])
    month_end = pd.Timestamp(selected_row["month_end"])
    validation_start = validation_start_date(history_index)
    trace = pd.DataFrame({"date": prices.index})
    trace["event_id"] = event_id
    trace["month_end"] = month_end
    trace["asset_i"] = asset_i
    trace["asset_j"] = asset_j
    trace["pair"] = f"{asset_i}/{asset_j}"
    trace["trade_rule"] = trading_config.trade_rule
    trace["model_split_period"] = split_period_values(prices.index, validation_start, month_end, trade_index)
    trace["event_phase"] = event_phase_values(prices.index, validation_start, month_end, trade_index)
    trace["asset_i_close"] = prices[asset_i].to_numpy(dtype=float)
    trace["asset_j_close"] = prices[asset_j].to_numpy(dtype=float)
    trace["asset_i_relative_price_pct"] = relative_price_pct(prices[asset_i])
    trace["asset_j_relative_price_pct"] = relative_price_pct(prices[asset_j])
    trace["spread"] = spread.reindex(prices.index).to_numpy(dtype=float)
    trace["spread_mean"] = spread_mean
    trace["spread_std"] = spread_std
    trace["spread_z"] = stable_zscore(trace["spread"], spread_mean, spread_std)
    trace["beta"] = beta
    trace["gamma"] = beta
    trace["intercept"] = intercept
    trace["predicted_sharpe"] = float(selected_row["predicted_sharpe"])
    trace["selection_rank"] = int(selected_row["rank"])
    return trace


def add_signal_return_columns(
    trace: pd.DataFrame,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    beta: float,
    config: TradingConfig,
) -> pd.DataFrame:
    """Add signal, trade action, and strategy return columns."""
    frame = trace.copy()
    date_index = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    signal = event_signal_series(date_index, signals)
    raw_return = raw_pair_daily_return(prices, asset_i, asset_j, beta).reindex(date_index).fillna(0.0)
    costs = signal.diff().abs().fillna(signal.abs()).multiply(config.transaction_cost_bps / 10_000.0)
    strategy_return = signal.shift(1).fillna(0.0).multiply(raw_return).subtract(costs)
    frame["signal"] = signal.to_numpy(dtype=int)
    frame["position_name"] = [position_name(value) for value in frame["signal"]]
    frame["trade_action"] = trade_action_values(signal)
    frame["trade_marker"] = trade_marker_values(frame["date"], trades)
    frame["raw_pair_daily_return"] = raw_return.to_numpy(dtype=float)
    frame["raw_pair_cumulative_return"] = raw_return.add(1.0).cumprod().subtract(1.0).to_numpy(dtype=float)
    frame["strategy_daily_return"] = strategy_return.to_numpy(dtype=float)
    frame["strategy_cumulative_return"] = strategy_return.add(1.0).cumprod().subtract(1.0).to_numpy(dtype=float)
    frame["transaction_cost"] = costs.to_numpy(dtype=float)
    frame["is_trade_date"] = frame["trade_marker"].astype(str).ne("")
    return frame


def add_threshold_columns(trace: pd.DataFrame, signals: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
    """Add trade-rule threshold columns across the full trace."""
    frame = trace.copy()
    if signals.empty:
        return frame
    for column in threshold_columns(config):
        if column in signals.columns:
            frame[column] = float(signals[column].dropna().iloc[0])
    return frame


def threshold_columns(config: TradingConfig) -> list[str]:
    """Return expected threshold columns for a trade rule."""
    if config.trade_rule == "vidyamurthy":
        return ["lower_band", "upper_band", "delta"]
    if config.trade_rule == "gatev":
        return ["long_entry", "short_entry", "long_exit", "short_exit"]
    return []


def event_signal_series(date_index: pd.DatetimeIndex, signals: pd.DataFrame) -> pd.Series:
    """Return signal values indexed by trace dates."""
    if signals.empty:
        return pd.Series(0, index=date_index, dtype=int)
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")["signal"].astype(int).reindex(date_index).fillna(0).astype(int)


def raw_pair_daily_return(prices: pd.DataFrame, asset_i: str, asset_j: str, beta: float) -> pd.Series:
    """Return unpositioned hedge-adjusted pair daily return."""
    simple_returns = prices[[asset_i, asset_j]].pct_change().fillna(0.0)
    values = simple_returns[asset_i].subtract(beta * simple_returns[asset_j]).divide(1.0 + abs(beta))
    return values.rename("raw_pair_daily_return")


def split_period_values(index: pd.Index, validation_start: pd.Timestamp, month_end: pd.Timestamp, trade_index: pd.Index) -> list[str]:
    """Return train, validation, and test labels."""
    trade_dates = set(pd.to_datetime(trade_index))
    values = []
    for date in pd.to_datetime(index):
        if date in trade_dates and date > month_end:
            values.append("test")
        elif date < validation_start:
            values.append("train")
        else:
            values.append("validation")
    return values


def event_phase_values(index: pd.Index, validation_start: pd.Timestamp, month_end: pd.Timestamp, trade_index: pd.Index) -> list[str]:
    """Return workflow phase labels."""
    trade_dates = set(pd.to_datetime(trade_index))
    values = []
    for date in pd.to_datetime(index):
        if date in trade_dates and date > month_end:
            values.append("trading_test")
        elif date == month_end:
            values.append("selection_date")
        elif date < validation_start:
            values.append("lookback_train")
        else:
            values.append("lookback_validation")
    return values


def validation_start_date(history_index: pd.Index) -> pd.Timestamp:
    """Return the chronological 70/30 split boundary for lookback diagnostics."""
    position = max(0, int(len(history_index) * 0.70))
    position = min(position, len(history_index) - 1)
    return pd.Timestamp(history_index[position])


def relative_price_pct(price: pd.Series) -> np.ndarray:
    """Return percent price movement from the first visible date."""
    base = float(price.iloc[0])
    if base == 0.0 or not np.isfinite(base):
        base = 1.0
    return price.divide(base).subtract(1.0).multiply(100.0).to_numpy(dtype=float)


def stable_zscore(values: pd.Series, mean: float, std: float) -> np.ndarray:
    """Return z-scores using stable spread statistics."""
    scale = std if std != 0.0 and np.isfinite(std) else 1.0
    return values.astype(float).subtract(mean).divide(scale).to_numpy(dtype=float)


def position_name(signal: int | float) -> str:
    """Return a readable position label."""
    if signal > 0:
        return "LONG_SPREAD"
    if signal < 0:
        return "SHORT_SPREAD"
    return "FLAT"


def trade_action_values(signal: pd.Series) -> list[str]:
    """Return open, close, and switch action labels from signal changes."""
    actions = []
    previous = 0
    for current in signal.astype(int):
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


def trade_marker_values(dates: pd.Series, trades: pd.DataFrame) -> list[str]:
    """Return trade episode open and close markers."""
    markers: dict[pd.Timestamp, list[str]] = {}
    if not trades.empty:
        for _, row in trades.iterrows():
            add_marker(markers, pd.Timestamp(row["open_date"]), f"open_{row['direction']}")
            add_marker(markers, pd.Timestamp(row["close_date"]), f"close_{row['direction']}")
    return [";".join(markers.get(pd.Timestamp(date), [])) for date in pd.to_datetime(dates)]


def add_marker(markers: dict[pd.Timestamp, list[str]], date: pd.Timestamp, value: str) -> None:
    """Add one marker value to a date."""
    markers.setdefault(date, []).append(value)


def ordered_trace_columns(trace: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
    """Return trace columns in a stable order."""
    leading = [
        "event_id",
        "month_end",
        "date",
        "model_split_period",
        "event_phase",
        "asset_i",
        "asset_j",
        "pair",
        "trade_rule",
        "predicted_sharpe",
        "selection_rank",
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
        "beta",
        "gamma",
        "intercept",
        "asset_i_close",
        "asset_j_close",
        "asset_i_relative_price_pct",
        "asset_j_relative_price_pct",
    ]
    thresholds = threshold_columns(config)
    remaining = [column for column in trace.columns if column not in leading + thresholds]
    return trace.loc[:, leading + thresholds + remaining]


def manifest_row(trace: pd.DataFrame, trades: pd.DataFrame, trace_path: Path, plot_path: Path) -> dict:
    """Return one manifest row."""
    first = trace.iloc[0]
    return {
        "event_id": first["event_id"],
        "month_end": first["month_end"],
        "asset_i": first["asset_i"],
        "asset_j": first["asset_j"],
        "pair": first["pair"],
        "trade_count": int(len(trades)),
        "total_strategy_return_pct": float(trace["strategy_cumulative_return"].iloc[-1] * 100.0),
        "trace_start_date": trace["date"].min(),
        "trace_end_date": trace["date"].max(),
        "trace_path": str(trace_path),
        "plot_path": str(plot_path),
    }


def save_pair_diagnostic_plot(trace: pd.DataFrame, trades: pd.DataFrame, image_path: Path) -> None:
    """Save one selected event's diagnostic plot."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    frame = trace.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    plot_price_panel(axes[0], frame, trades)
    plot_spread_panel(axes[1], frame)
    plot_signal_panel(axes[2], frame)
    plot_return_panel(axes[3], frame)
    fig.suptitle(f"{frame['event_id'].iloc[0]} | {frame['pair'].iloc[0]}", fontsize=13)
    fig.savefig(image_path, dpi=150)
    plt.close(fig)


def plot_price_panel(axis, trace: pd.DataFrame, trades: pd.DataFrame) -> None:
    """Plot relative price movement."""
    apply_split_background(axis, trace)
    asset_i = str(trace["asset_i"].iloc[0])
    asset_j = str(trace["asset_j"].iloc[0])
    axis.plot(trace["date"], trace["asset_i_relative_price_pct"], label=f"{asset_i} relative %", linewidth=1.4)
    axis.plot(trace["date"], trace["asset_j_relative_price_pct"], label=f"{asset_j} relative %", linewidth=1.4)
    mark_boundaries(axis, trace)
    mark_trade_dates(axis, trades)
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    axis.set_ylabel("Relative price (%)")
    axis.legend(loc="upper left", ncols=2)
    axis.grid(alpha=0.25)


def plot_spread_panel(axis, trace: pd.DataFrame) -> None:
    """Plot spread and trade thresholds."""
    apply_split_background(axis, trace)
    axis.plot(trace["date"], trace["spread"], label="spread", color="#1f2937", linewidth=1.2)
    axis.plot(trace["date"], trace["spread_mean"], label="spread_mean", color="#6b7280", linewidth=0.9)
    for column in plot_threshold_columns(trace):
        axis.plot(trace["date"], trace[column], label=column, linewidth=0.9)
    plot_action_markers(axis, trace)
    mark_boundaries(axis, trace)
    axis.set_ylabel("Spread")
    axis.legend(loc="upper left", ncols=3)
    axis.grid(alpha=0.25)


def plot_signal_panel(axis, trace: pd.DataFrame) -> None:
    """Plot signal position."""
    apply_split_background(axis, trace)
    axis.step(trace["date"], trace["signal"], where="post", label="signal", linewidth=1.2)
    mark_boundaries(axis, trace)
    axis.set_ylabel("Signal")
    axis.set_yticks([-1, 0, 1])
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)


def plot_return_panel(axis, trace: pd.DataFrame) -> None:
    """Plot daily and cumulative returns."""
    apply_split_background(axis, trace)
    axis.bar(trace["date"], trace["strategy_daily_return"].multiply(100.0), label="strategy daily return %", color="#94a3b8", width=1.0)
    twin = axis.twinx()
    plot_cumulative_return_line(
        twin,
        trace,
        "strategy_cumulative_return",
        "strategy cumulative return %",
        "#0f766e",
        1.4,
    )
    mark_boundaries(axis, trace)
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    axis.set_ylabel("Daily return (%)")
    twin.set_ylabel("Cumulative return (%)")
    merge_legends(axis, twin)
    axis.grid(alpha=0.25)


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


def plot_threshold_columns(trace: pd.DataFrame) -> list[str]:
    """Return threshold columns present in a trace."""
    candidates = ["lower_band", "upper_band", "long_entry", "short_entry", "long_exit", "short_exit"]
    return [column for column in candidates if column in trace.columns]


def plot_action_markers(axis, trace: pd.DataFrame) -> None:
    """Plot long and short signal action markers."""
    long_rows = trace[trace["trade_action"].str.contains("long", na=False)]
    short_rows = trace[trace["trade_action"].str.contains("short", na=False)]
    if not long_rows.empty:
        axis.scatter(long_rows["date"], long_rows["spread"], marker="^", color="#16a34a", s=26, label="long action", zorder=5)
    if not short_rows.empty:
        axis.scatter(short_rows["date"], short_rows["spread"], marker="v", color="#dc2626", s=26, label="short action", zorder=5)


def apply_split_background(axis, trace: pd.DataFrame) -> None:
    """Shade train, validation, and test periods."""
    for split_name, group in trace.groupby("model_split_period", sort=False):
        axis.axvspan(pd.to_datetime(group["date"]).min(), pd.to_datetime(group["date"]).max(), color=split_color(str(split_name)), alpha=0.10, linewidth=0)


def split_color(split_name: str) -> str:
    """Return a background color for a split label."""
    if split_name == "train":
        return "#16a34a"
    if split_name == "validation":
        return "#ca8a04"
    return "#2563eb"


def mark_boundaries(axis, trace: pd.DataFrame) -> None:
    """Mark validation and test boundaries."""
    validation_dates = trace.loc[trace["model_split_period"].eq("validation"), "date"]
    test_dates = trace.loc[trace["model_split_period"].eq("test"), "date"]
    if not validation_dates.empty:
        axis.axvline(pd.to_datetime(validation_dates).min(), color="#92400e", linewidth=1.0, linestyle="--", alpha=0.75)
    if not test_dates.empty:
        axis.axvline(pd.to_datetime(test_dates).min(), color="black", linewidth=1.0, linestyle="--", alpha=0.75)


def mark_trade_dates(axis, trades: pd.DataFrame) -> None:
    """Mark trade open and close dates."""
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


def reset_directory(directory: Path, suffix: str) -> None:
    """Remove stale diagnostic files from a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix == suffix:
            path.unlink()
