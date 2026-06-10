"""Pair-level relative price movement plots."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_pair_price_movement_plot(
    pair_key: str,
    asset_y: str,
    asset_x: str,
    price_y: pd.Series,
    price_x: pd.Series,
    standard_signals: pd.DataFrame,
    forecast_signals: pd.DataFrame,
    validation_start: pd.Timestamp,
    trading_start: pd.Timestamp,
    output_path: Path,
) -> None:
    """Save relative price moves with period splits and entry timing markers."""
    movement = _relative_price_movement(asset_y, asset_x, price_y, price_x)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (price_ax, timing_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 6.2),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.0], "hspace": 0.08},
    )
    _plot_relative_prices(price_ax, pair_key, asset_y, asset_x, movement, validation_start, trading_start)
    _plot_entry_timing_panel(timing_ax, movement.index, standard_signals, forecast_signals, validation_start, trading_start)
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relative_prices(
    ax,
    pair_key: str,
    asset_y: str,
    asset_x: str,
    movement: pd.DataFrame,
    validation_start: pd.Timestamp,
    trading_start: pd.Timestamp,
) -> None:
    _add_period_backgrounds(ax, movement.index, validation_start, trading_start, include_labels=True)
    ax.plot(movement.index, movement[asset_y], label=asset_y.upper(), linewidth=1.6)
    ax.plot(movement.index, movement[asset_x], label=asset_x.upper(), linewidth=1.6)
    ax.axhline(0.0, color="#1f2937", linewidth=0.8, alpha=0.55)
    _add_period_boundaries(ax, validation_start, trading_start, include_labels=True)
    ax.set_title(f"{pair_key}: relative price movement")
    ax.set_ylabel("Relative move from first bar (%)")
    ax.grid(True, linewidth=0.5, alpha=0.28)
    ax.legend(loc="best")


def _plot_entry_timing_panel(
    ax,
    index: pd.Index,
    standard_signals: pd.DataFrame,
    forecast_signals: pd.DataFrame,
    validation_start: pd.Timestamp,
    trading_start: pd.Timestamp,
) -> None:
    _add_period_backgrounds(ax, index, validation_start, trading_start, include_labels=False)
    _add_period_boundaries(ax, validation_start, trading_start, include_labels=False)
    _plot_entry_markers(ax, standard_signals, y_value=1.0, label_prefix="standard")
    _plot_entry_markers(ax, forecast_signals, y_value=0.0, label_prefix="forecast")
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["standard", "forecast"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Date")
    ax.grid(True, axis="x", linewidth=0.5, alpha=0.28)
    ax.legend(loc="upper left", ncols=2)


def _plot_entry_markers(ax, signals: pd.DataFrame, y_value: float, label_prefix: str) -> None:
    long_entries, short_entries = _entry_timestamps(signals)
    if not long_entries.empty:
        ax.scatter(
            long_entries,
            [y_value] * len(long_entries),
            marker="^",
            color="#16a34a",
            s=44,
            label=f"{label_prefix} long",
            zorder=4,
        )
    if not short_entries.empty:
        ax.scatter(
            short_entries,
            [y_value] * len(short_entries),
            marker="v",
            color="#dc2626",
            s=44,
            label=f"{label_prefix} short",
            zorder=4,
        )


def _entry_timestamps(signals: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    frame = _clean_signal_frame(signals)
    if frame.empty:
        return pd.Series(dtype="datetime64[ns]"), pd.Series(dtype="datetime64[ns]")
    previous = frame["signal"].shift(1).fillna(0).astype(int)
    current = frame["signal"].astype(int)
    long_entries = frame.loc[(current == 1) & (previous != 1), "datetime"]
    short_entries = frame.loc[(current == -1) & (previous != -1), "datetime"]
    return long_entries, short_entries


def _clean_signal_frame(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    frame = signals.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["signal"] = frame["signal"].astype(int)
    return frame.dropna(subset=["datetime", "signal"])


def _add_period_backgrounds(
    ax,
    index: pd.Index,
    validation_start: pd.Timestamp,
    trading_start: pd.Timestamp,
    include_labels: bool,
) -> None:
    start = pd.Timestamp(index[0])
    end = pd.Timestamp(index[-1])
    validation_start = pd.Timestamp(validation_start)
    trading_start = pd.Timestamp(trading_start)
    labels = ("training", "validation", "trading") if include_labels else (None, None, None)
    ax.axvspan(start, validation_start, color="#dbeafe", alpha=0.18, linewidth=0, label=labels[0])
    ax.axvspan(validation_start, trading_start, color="#fef3c7", alpha=0.22, linewidth=0, label=labels[1])
    ax.axvspan(trading_start, end, color="#dcfce7", alpha=0.18, linewidth=0, label=labels[2])


def _add_period_boundaries(
    ax,
    validation_start: pd.Timestamp,
    trading_start: pd.Timestamp,
    include_labels: bool,
) -> None:
    ax.axvline(
        pd.Timestamp(validation_start),
        color="#92400e",
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        label="train/validation split" if include_labels else None,
    )
    ax.axvline(
        pd.Timestamp(trading_start),
        color="#111827",
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        label="validation/trading split" if include_labels else None,
    )


def _relative_price_movement(
    asset_y: str,
    asset_x: str,
    price_y: pd.Series,
    price_x: pd.Series,
) -> pd.DataFrame:
    prices = pd.concat(
        [price_y.astype(float).rename(asset_y), price_x.astype(float).rename(asset_x)],
        axis=1,
    ).dropna()
    if prices.empty:
        raise ValueError("Pair price movement plot requires non-empty prices.")
    return prices.divide(prices.iloc[0]).subtract(1.0).multiply(100.0)
