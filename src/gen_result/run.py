"""
Performance report generator for PairsTradingQFin05 signals.

Usage:
    uv run src/gen_result/run.py

Outputs (data/results/PairsTradingQFin05/standard/kalman_em/):
    {pair1}_{pair2}_{interval}_report.txt   metrics summary
    {pair1}_{pair2}_{interval}_return.png   cumulative-return chart
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from src.gen_result.metrics import (
    compute_daily_returns,
    compute_benchmark_returns,
    build_report,
)
from src.gen_result.plot import plot_performance

# ── 設定 ──────────────────────────────────────────────────────────────────
PAIR1    = "spy"
PAIR2    = "qqq"
INTERVAL = "1d"

DATA_DIR    = _ROOT / "data"
SIGNALS_DIR = DATA_DIR / "signals" / "PairsTradingQFin05" / "standard" / "kalman_em"
RAW_DIR     = DATA_DIR / "raw"
OUTPUT_DIR  = DATA_DIR / "results" / "PairsTradingQFin05" / "standard" / "kalman_em"
# ──────────────────────────────────────────────────────────────────────────


def load_prices(pair: str) -> pd.Series:
    path = RAW_DIR / pair / f"{INTERVAL}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df["close"].dropna().astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def load_signals(pair1: str, pair2: str) -> pd.DataFrame:
    path = SIGNALS_DIR / f"{pair1}_{pair2}_{INTERVAL}_signals.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df


def print_report(report: dict, pair1: str, pair2: str) -> str:
    lines = [
        "=" * 52,
        f"  Pairs Trading Report: {pair1.upper()} / {pair2.upper()}",
        f"  Model: Elliott, Van der Hoek & Malcolm (2005)",
        "=" * 52,
        "",
        "  --- Strategy ---",
        f"  Total return          : {report['total_return_%']:>8.2f} %",
        f"  Annualised return     : {report['annualised_return_%']:>8.2f} %",
        f"  Sharpe ratio          : {report['sharpe_ratio']:>8.3f}",
        f"  Max drawdown          : {report['max_drawdown_%']:>8.2f} %",
        f"  Win rate (active days): {report['win_rate_%']:>8.2f} %",
        f"  Active days / Total   : {report['active_days']} / {report['total_days']}",
    ]
    if "benchmark_total_return_%" in report:
        lines += [
            "",
            f"  --- Benchmark ({pair1.upper()} buy & hold) ---",
            f"  Total return          : {report['benchmark_total_return_%']:>8.2f} %",
            f"  Annualised return     : {report['benchmark_annualised_return_%']:>8.2f} %",
            f"  Sharpe ratio          : {report['benchmark_sharpe_ratio']:>8.3f}",
            f"  Max drawdown          : {report['benchmark_max_drawdown_%']:>8.2f} %",
        ]
    lines.append("=" * 52)
    text = "\n".join(lines)
    print(text)
    return text


def main() -> None:
    print(f"[LOAD] signals: {PAIR1}/{PAIR2}  interval={INTERVAL}")
    signals_df = load_signals(PAIR1, PAIR2)

    print(f"[LOAD] prices: {PAIR1}, {PAIR2}")
    price1 = load_prices(PAIR1)
    price2 = load_prices(PAIR2)

    # Trim prices to the signal period
    start = pd.to_datetime(signals_df["datetime"].min())
    price1 = price1[price1.index >= start]
    price2 = price2[price2.index >= start]

    # Compute returns
    daily_ret  = compute_daily_returns(signals_df, price1, price2)
    bench_ret  = compute_benchmark_returns(price1)

    # Metrics
    report = build_report(daily_ret, bench_ret)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{PAIR1}_{PAIR2}_{INTERVAL}"

    # Print & save text report
    report_text = print_report(report, PAIR1, PAIR2)
    report_path = OUTPUT_DIR / f"{stem}_report.txt"
    report_path.write_text(report_text)
    print(f"[SAVE] {report_path}")

    # Plot
    plot_path = OUTPUT_DIR / f"{stem}_return.png"
    plot_performance(daily_ret, bench_ret, PAIR1, PAIR2, plot_path)


if __name__ == "__main__":
    main()
