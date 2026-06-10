"""Per-pair daily diagnostics for the Enhancing ML pair-trading workflow."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from common.models import PairCandidate


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def prepare_pair_diagnostic_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Reset diagnostic trace and plot directories for a run."""
    trace_dir = output_dir / "pair_daily_traces"
    plot_dir = output_dir / "pair_diagnostic_plots"
    reset_directory(trace_dir, ".csv")
    reset_directory(plot_dir, ".png")
    return trace_dir, plot_dir


def save_pair_diagnostic_outputs(
    pair_key: str,
    pair: PairCandidate,
    prices: pd.DataFrame,
    full_spread: pd.Series,
    standard_signals: pd.DataFrame,
    standard_trades: pd.DataFrame,
    forecast_signals: pd.DataFrame,
    forecast_trades: pd.DataFrame,
    validation_start_date: pd.Timestamp,
    trading_start_date: pd.Timestamp,
    forecast_model: str,
    trace_dir: Path,
    plot_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Save one pair's daily trace CSV and diagnostic plot."""
    trace = build_pair_daily_trace(
        pair_key,
        pair,
        prices,
        full_spread,
        standard_signals,
        standard_trades,
        forecast_signals,
        forecast_trades,
        validation_start_date,
        trading_start_date,
        forecast_model,
    )
    trace_path = trace_dir / f"{pair_key}.csv"
    plot_path = plot_dir / f"{pair_key}.png"
    trace.to_csv(trace_path, index=False)
    save_pair_diagnostic_plot(trace, pair_key, forecast_model, plot_path)
    return trace, manifest_row(pair_key, pair, trace, trace_path, plot_path)


def save_pair_diagnostic_summary(frames: list[pd.DataFrame], manifest_rows: list[dict], output_dir: Path) -> None:
    """Save combined diagnostic CSV and manifest files."""
    combined = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    combined.to_csv(output_dir / "pair_daily_traces.csv", index=False)
    manifest.to_csv(output_dir / "pair_daily_trace_manifest.csv", index=False)


def build_pair_daily_trace(
    pair_key: str,
    pair: PairCandidate,
    prices: pd.DataFrame,
    full_spread: pd.Series,
    standard_signals: pd.DataFrame,
    standard_trades: pd.DataFrame,
    forecast_signals: pd.DataFrame,
    forecast_trades: pd.DataFrame,
    validation_start_date: pd.Timestamp,
    trading_start_date: pd.Timestamp,
    forecast_model: str,
) -> pd.DataFrame:
    """Build one pair's train, validation, and test daily diagnostics."""
    asset_y = pair.asset_y
    asset_x = pair.asset_x
    visible_prices = prices.loc[full_spread.index, [asset_y, asset_x]].astype(float).dropna()
    trace = pd.DataFrame({"date": visible_prices.index})
    trace["pair"] = pair_key
    trace["asset_y"] = asset_y
    trace["asset_x"] = asset_x
    trace["model_split_period"] = split_period_values(visible_prices.index, validation_start_date, trading_start_date)
    trace["event_phase"] = event_phase_values(visible_prices.index, validation_start_date, trading_start_date)
    trace["asset_y_close"] = visible_prices[asset_y].to_numpy(dtype=float)
    trace["asset_x_close"] = visible_prices[asset_x].to_numpy(dtype=float)
    trace["asset_y_relative_price_pct"] = relative_price_pct(visible_prices[asset_y])
    trace["asset_x_relative_price_pct"] = relative_price_pct(visible_prices[asset_x])
    trace["spread"] = full_spread.reindex(visible_prices.index).to_numpy(dtype=float)
    trace["hedge_ratio"] = pair.hedge_ratio
    trace["intercept"] = pair.intercept
    trace["adf_t_stat"] = pair.diagnostics.adf_t_stat
    trace["hurst"] = pair.diagnostics.hurst
    trace["half_life_bars"] = pair.diagnostics.half_life_bars
    raw_return = raw_pair_daily_return(visible_prices, asset_y, asset_x)
    trace["raw_pair_daily_return"] = raw_return.reindex(visible_prices.index).fillna(0.0).to_numpy(dtype=float)
    trace["raw_pair_cumulative_return"] = raw_return.add(1.0).cumprod().subtract(1.0).reindex(visible_prices.index).fillna(0.0).to_numpy(dtype=float)
    trace = add_strategy_columns(trace, standard_signals, standard_trades, "standard")
    trace = add_strategy_columns(trace, forecast_signals, forecast_trades, "forecast")
    trace["forecast_model"] = forecast_model
    return ordered_trace_columns(trace)


def add_strategy_columns(trace: pd.DataFrame, signals: pd.DataFrame, trades: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add signal, threshold, and return columns for one strategy variant."""
    frame = trace.copy()
    signal_frame = normalize_signal_dates(signals)
    date_index = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    signal = signal_frame.set_index("date")["signal"].astype(float).reindex(date_index).fillna(0.0)
    strategy_return = signal.shift(1).fillna(0.0).multiply(frame.set_index("date")["raw_pair_daily_return"].reindex(date_index).fillna(0.0))
    frame[f"{prefix}_signal"] = signal.to_numpy(dtype=int)
    frame[f"{prefix}_position_name"] = [position_name(value) for value in frame[f"{prefix}_signal"]]
    frame[f"{prefix}_trade_action"] = trade_action_values(signal)
    frame[f"{prefix}_trade_marker"] = trade_marker_values(frame["date"], trades)
    frame[f"{prefix}_daily_return"] = strategy_return.to_numpy(dtype=float)
    frame[f"{prefix}_cumulative_return"] = strategy_return.add(1.0).cumprod().subtract(1.0).to_numpy(dtype=float)
    frame = add_signal_columns(frame, signal_frame, prefix, date_index)
    return frame


def add_signal_columns(frame: pd.DataFrame, signal_frame: pd.DataFrame, prefix: str, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Add non-signal diagnostic columns from a signal frame."""
    skip_columns = {"datetime", "date", "signal", "spread"}
    for column in signal_frame.columns:
        if column in skip_columns:
            continue
        values = signal_frame.set_index("date")[column].reindex(date_index)
        frame[f"{prefix}_{column}"] = values.to_numpy()
    return frame


def normalize_signal_dates(signals: pd.DataFrame) -> pd.DataFrame:
    """Return a signal frame with a date column."""
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["datetime"])
    return frame


def raw_pair_daily_return(prices: pd.DataFrame, asset_y: str, asset_x: str) -> pd.Series:
    """Return the unpositioned equal-dollar pair daily return."""
    returns_y = prices[asset_y].pct_change()
    returns_x = prices[asset_x].pct_change()
    return (0.5 * returns_y - 0.5 * returns_x).fillna(0.0).rename("raw_pair_daily_return")


def split_period_values(index: pd.Index, validation_start_date: pd.Timestamp, trading_start_date: pd.Timestamp) -> list[str]:
    """Return train, validation, and test labels for each date."""
    values = []
    validation_start = pd.Timestamp(validation_start_date)
    trading_start = pd.Timestamp(trading_start_date)
    for date in pd.to_datetime(index):
        if date < validation_start:
            values.append("train")
        elif date < trading_start:
            values.append("validation")
        else:
            values.append("test")
    return values


def event_phase_values(index: pd.Index, validation_start_date: pd.Timestamp, trading_start_date: pd.Timestamp) -> list[str]:
    """Return workflow phase labels for each date."""
    values = []
    validation_start = pd.Timestamp(validation_start_date)
    trading_start = pd.Timestamp(trading_start_date)
    for date in pd.to_datetime(index):
        if date < validation_start:
            values.append("formation_train")
        elif date < trading_start:
            values.append("formation_validation")
        else:
            values.append("trading_test")
    return values


def relative_price_pct(price: pd.Series) -> np.ndarray:
    """Return percent price movement from the first visible date."""
    base = float(price.iloc[0])
    if base == 0.0 or not np.isfinite(base):
        base = 1.0
    return price.divide(base).subtract(1.0).multiply(100.0).to_numpy(dtype=float)


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
    """Return one action label for a signal transition."""
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
    """Return trade open and close markers for trace dates."""
    markers: dict[pd.Timestamp, list[str]] = {}
    if not trades.empty:
        frame = trades.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        for _, row in frame.iterrows():
            add_marker(markers, pd.Timestamp(row["datetime"]), f"{row['action']}_{row['direction']}")
    return [";".join(markers.get(pd.Timestamp(date), [])) for date in pd.to_datetime(dates)]


def add_marker(markers: dict[pd.Timestamp, list[str]], date: pd.Timestamp, value: str) -> None:
    """Add one marker value to a date."""
    markers.setdefault(date, []).append(value)


def ordered_trace_columns(trace: pd.DataFrame) -> pd.DataFrame:
    """Return trace columns in a stable order."""
    leading = [
        "pair",
        "date",
        "model_split_period",
        "event_phase",
        "asset_y",
        "asset_x",
        "forecast_model",
        "asset_y_close",
        "asset_x_close",
        "asset_y_relative_price_pct",
        "asset_x_relative_price_pct",
        "spread",
        "raw_pair_daily_return",
        "raw_pair_cumulative_return",
        "standard_signal",
        "standard_position_name",
        "standard_trade_action",
        "standard_trade_marker",
        "standard_daily_return",
        "standard_cumulative_return",
        "forecast_signal",
        "forecast_position_name",
        "forecast_trade_action",
        "forecast_trade_marker",
        "forecast_daily_return",
        "forecast_cumulative_return",
        "hedge_ratio",
        "intercept",
        "adf_t_stat",
        "hurst",
        "half_life_bars",
    ]
    remaining = [column for column in trace.columns if column not in leading]
    return trace.loc[:, leading + remaining]


def manifest_row(pair_key: str, pair: PairCandidate, trace: pd.DataFrame, trace_path: Path, plot_path: Path) -> dict:
    """Return one manifest row."""
    return {
        "pair": pair_key,
        "asset_y": pair.asset_y,
        "asset_x": pair.asset_x,
        "trace_start_date": trace["date"].min(),
        "trace_end_date": trace["date"].max(),
        "standard_total_return_pct": float(trace["standard_cumulative_return"].iloc[-1] * 100.0),
        "forecast_total_return_pct": float(trace["forecast_cumulative_return"].iloc[-1] * 100.0),
        "trace_path": str(trace_path),
        "plot_path": str(plot_path),
    }


def save_pair_diagnostic_plot(trace: pd.DataFrame, pair_key: str, forecast_model: str, image_path: Path) -> None:
    """Save one pair's diagnostic plot."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    frame = trace.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True, constrained_layout=True)
    plot_price_panel(axes[0], frame)
    plot_standard_spread_panel(axes[1], frame)
    plot_forecast_indicator_panel(axes[2], frame, forecast_model)
    plot_signal_panel(axes[3], frame)
    plot_return_panel(axes[4], frame)
    fig.suptitle(f"{pair_key} | forecast_{forecast_model}", fontsize=13)
    fig.savefig(image_path, dpi=150)
    plt.close(fig)


def plot_price_panel(axis, trace: pd.DataFrame) -> None:
    """Plot relative price movement."""
    apply_split_background(axis, trace)
    asset_y = str(trace["asset_y"].iloc[0])
    asset_x = str(trace["asset_x"].iloc[0])
    axis.plot(trace["date"], trace["asset_y_relative_price_pct"], label=f"{asset_y} relative %", linewidth=1.4)
    axis.plot(trace["date"], trace["asset_x_relative_price_pct"], label=f"{asset_x} relative %", linewidth=1.4)
    mark_boundaries(axis, trace)
    axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    axis.set_ylabel("Relative price (%)")
    axis.legend(loc="upper left", ncols=2)
    axis.grid(alpha=0.25)


def plot_standard_spread_panel(axis, trace: pd.DataFrame) -> None:
    """Plot spread and standard threshold levels."""
    apply_split_background(axis, trace)
    axis.plot(trace["date"], trace["spread"], label="spread", color="#1f2937", linewidth=1.2)
    for column in ("standard_long_entry", "standard_short_entry", "standard_exit"):
        if column in trace.columns:
            axis.plot(trace["date"], trace[column], label=column.replace("standard_", ""), linewidth=0.9)
    plot_action_markers(axis, trace, "standard")
    mark_boundaries(axis, trace)
    axis.set_ylabel("Spread")
    axis.legend(loc="upper left", ncols=4)
    axis.grid(alpha=0.25)


def plot_forecast_indicator_panel(axis, trace: pd.DataFrame, forecast_model: str) -> None:
    """Plot forecast trading indicators."""
    apply_split_background(axis, trace)
    if "forecast_predicted_change_pct" in trace.columns:
        axis.plot(trace["date"], trace["forecast_predicted_change_pct"], label="predicted_change_pct", color="#0f766e", linewidth=1.2)
        for column in ("forecast_long_entry_pct", "forecast_short_entry_pct"):
            if column in trace.columns:
                axis.plot(trace["date"], trace[column], label=column.replace("forecast_", ""), linewidth=0.9)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
        axis.set_ylabel("Pred change (%)")
    elif "forecast_predicted_spread" in trace.columns:
        axis.plot(trace["date"], trace["spread"], label="spread", color="#1f2937", linewidth=1.0)
        axis.plot(trace["date"], trace["forecast_predicted_spread"], label="predicted_spread", color="#0f766e", linewidth=1.2)
        axis.set_ylabel("Spread")
    else:
        axis.plot(trace["date"], trace["forecast_signal"], label=f"forecast_{forecast_model}_signal", color="#0f766e", linewidth=1.0)
        axis.set_ylabel("Signal")
    plot_action_markers(axis, trace, "forecast")
    mark_boundaries(axis, trace)
    axis.legend(loc="upper left", ncols=3)
    axis.grid(alpha=0.25)


def plot_signal_panel(axis, trace: pd.DataFrame) -> None:
    """Plot standard and forecast signals."""
    apply_split_background(axis, trace)
    axis.step(trace["date"], trace["standard_signal"], where="post", label="standard signal", linewidth=1.1)
    axis.step(trace["date"], trace["forecast_signal"], where="post", label="forecast signal", linewidth=1.1)
    mark_boundaries(axis, trace)
    axis.set_ylabel("Signal")
    axis.set_yticks([-1, 0, 1])
    axis.legend(loc="upper left", ncols=2)
    axis.grid(alpha=0.25)


def plot_return_panel(axis, trace: pd.DataFrame) -> None:
    """Plot daily and cumulative strategy returns."""
    apply_split_background(axis, trace)
    axis.bar(trace["date"], trace["standard_daily_return"].multiply(100.0), label="standard daily %", color="#94a3b8", width=1.0)
    axis.bar(trace["date"], trace["forecast_daily_return"].multiply(100.0), label="forecast daily %", color="#f59e0b", width=1.0, alpha=0.45)
    twin = axis.twinx()
    plot_cumulative_return_line(twin, trace, "standard_cumulative_return", "standard cumulative %", "#2563eb", 1.3)
    plot_cumulative_return_line(twin, trace, "forecast_cumulative_return", "forecast cumulative %", "#0f766e", 1.3)
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


def plot_action_markers(axis, trace: pd.DataFrame, prefix: str) -> None:
    """Plot long and short action markers on a spread-like axis."""
    action_column = f"{prefix}_trade_action"
    if action_column not in trace.columns:
        return
    long_rows = trace[trace[action_column].str.contains("long", na=False)]
    short_rows = trace[trace[action_column].str.contains("short", na=False)]
    if not long_rows.empty:
        axis.scatter(long_rows["date"], long_rows["spread"], marker="^", color="#16a34a", s=24, label=f"{prefix} long action", zorder=5)
    if not short_rows.empty:
        axis.scatter(short_rows["date"], short_rows["spread"], marker="v", color="#dc2626", s=24, label=f"{prefix} short action", zorder=5)


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


def merge_legends(axis, twin_axis) -> None:
    """Merge legends from two y-axes."""
    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin_axis.get_legend_handles_labels()
    axis.legend(handles + twin_handles, labels + twin_labels, loc="upper left", ncols=2)


def reset_directory(directory: Path, suffix: str) -> None:
    """Remove stale files from a diagnostic directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix == suffix:
            path.unlink()
