"""Pair-level trade timing plots."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_trade_timing_plot(
    pair_key: str,
    model_name: str,
    signals: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a spread plot annotated with long, short, and close timing."""
    clean_signals = _clean_signal_frame(signals)
    if clean_signals.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    ax.plot(clean_signals["datetime"], clean_signals["spread"], label="spread", color="#111827", linewidth=1.25)
    if "predicted_spread" in clean_signals.columns:
        ax.plot(
            clean_signals["datetime"],
            clean_signals["predicted_spread"],
            label="predicted spread",
            color="#2563eb",
            linewidth=1.05,
            linestyle="--",
            alpha=0.78,
        )

    _shade_positions(ax, clean_signals)
    _plot_transition_markers(ax, clean_signals)
    ax.set_title(f"{pair_key}: {model_name} trade timing")
    ax.set_ylabel("Spread")
    ax.set_xlabel("Date")
    ax.grid(True, linewidth=0.5, alpha=0.28)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _clean_signal_frame(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    frame = signals.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["signal"] = frame["signal"].astype(int)
    frame["spread"] = frame["spread"].astype(float)
    if "predicted_spread" in frame.columns:
        frame["predicted_spread"] = frame["predicted_spread"].astype(float)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["datetime", "spread", "signal"])


def _shade_positions(ax, signals: pd.DataFrame) -> None:
    timestamps = signals["datetime"].to_list()
    positions = signals["signal"].to_numpy(dtype=int)
    for idx, position in enumerate(positions):
        if position == 0:
            continue
        start = timestamps[idx]
        end = timestamps[idx + 1] if idx + 1 < len(timestamps) else timestamps[idx]
        color = "#16a34a" if position > 0 else "#dc2626"
        ax.axvspan(start, end, color=color, alpha=0.08, linewidth=0)


def _plot_transition_markers(ax, signals: pd.DataFrame) -> None:
    previous = signals["signal"].shift(1).fillna(0).astype(int)
    current = signals["signal"].astype(int)
    long_open = signals[(current == 1) & (previous != 1)]
    short_open = signals[(current == -1) & (previous != -1)]
    close = signals[(current == 0) & (previous != 0)]

    ax.scatter(long_open["datetime"], long_open["spread"], marker="^", color="#16a34a", s=46, label="long open", zorder=4)
    ax.scatter(short_open["datetime"], short_open["spread"], marker="v", color="#dc2626", s=46, label="short open", zorder=4)
    ax.scatter(close["datetime"], close["spread"], marker="x", color="#111827", s=38, label="close", zorder=4)
