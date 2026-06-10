"""
Visualisation for pairs trading performance.

Generates a single figure with three panels:
  1. Cumulative return — strategy vs benchmark buy-and-hold
  2. Drawdown — underwater plot for the strategy
  3. Rolling 63-day Sharpe ratio — tracks how alpha varies over time
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def _cum_ret(daily_ret: pd.Series) -> pd.Series:
    cumulative = (1 + daily_ret).cumprod() - 1
    return _with_zero_baseline(cumulative)


def _with_zero_baseline(series: pd.Series) -> pd.Series:
    """Return a plotting series that explicitly starts from 0%."""
    if series.empty:
        return series
    baseline_index = _baseline_timestamp(series.index)
    baseline = pd.Series([0.0], index=pd.DatetimeIndex([baseline_index]), name=series.name)
    return pd.concat([baseline, series])


def _baseline_timestamp(index: pd.Index) -> pd.Timestamp:
    """Return the timestamp used for the left-edge cumulative-return baseline."""
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if len(dates) > 1:
        first_gap = dates[1] - dates[0]
        if first_gap > pd.Timedelta(0):
            return pd.Timestamp(dates[0] - first_gap)
    return pd.Timestamp(dates[0] - pd.Timedelta(days=1))


def _drawdown(daily_ret: pd.Series) -> pd.Series:
    cum = (1 + daily_ret).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def _rolling_sharpe(daily_ret: pd.Series, window: int = 63) -> pd.Series:
    """Rolling annualised Sharpe over `window` trading days."""
    roll_mean = daily_ret.rolling(window).mean()
    roll_std  = daily_ret.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).rename("rolling_sharpe")


def plot_performance(
    daily_ret: pd.Series,
    benchmark_ret: pd.Series,
    pair1: str,
    pair2: str,
    output_path: Path,
) -> None:
    """Save a three-panel performance chart to output_path."""

    # Align benchmark to strategy dates
    bench = benchmark_ret.reindex(daily_ret.index).fillna(0)

    cum_strat = _cum_ret(daily_ret) * 100          # → %
    cum_bench = _cum_ret(bench) * 100
    dd_strat  = _drawdown(daily_ret) * 100
    roll_sr   = _rolling_sharpe(daily_ret)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle(
        f"Pairs Trading: {pair1.upper()} / {pair2.upper()}  "
        f"(Elliott et al. 2005)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Panel 1: Cumulative return ──────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(cum_strat.index, cum_strat.values,
             color="#1f77b4", linewidth=1.2, label="Strategy")
    ax1.plot(cum_bench.index, cum_bench.values,
             color="#ff7f0e", linewidth=1.0, linestyle="--",
             label=f"{pair1.upper()} buy & hold")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax1.grid(alpha=0.3)

    final_s = cum_strat.iloc[-1]
    final_b = cum_bench.iloc[-1]
    ax1.annotate(f"{final_s:+.1f}%", xy=(cum_strat.index[-1], final_s),
                 xytext=(5, 0), textcoords="offset points",
                 color="#1f77b4", fontsize=8, va="center")
    ax1.annotate(f"{final_b:+.1f}%", xy=(cum_bench.index[-1], final_b),
                 xytext=(5, 0), textcoords="offset points",
                 color="#ff7f0e", fontsize=8, va="center")

    # ── Panel 2: Drawdown ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(dd_strat.index, dd_strat.values, 0,
                     color="#d62728", alpha=0.55, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Rolling Sharpe ─────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(roll_sr.index, roll_sr.values,
             color="#2ca02c", linewidth=1.0, label="Rolling Sharpe (63d)")
    ax3.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax3.axhline(1, color="#2ca02c", linewidth=0.5, linestyle="--", alpha=0.6)
    ax3.set_ylabel("Sharpe (63d)", fontsize=10)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(alpha=0.3)

    # X-axis formatting
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {output_path}")
