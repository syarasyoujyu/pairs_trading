"""
Elliott et al. (2005) pairs trading — entry point.

Usage:
    uv run src/paper_methods/PairsTradingQFin05/run.py

Outputs (under data/signals/PairsTradingQFin05/standard/kalman_em/):
    {pair1}_{pair2}_{interval}_signals.csv   per-bar spread & z-score
    {pair1}_{pair2}_{interval}_trades.csv    entry / exit events with lots
"""
import sys
from pathlib import Path

# Make the project root importable regardless of working directory
_ROOT = Path(__file__).parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from src.paper_methods.PairsTradingQFin05.spread import build_spread
from src.paper_methods.PairsTradingQFin05.signals import generate_signals

# ── 設定 ──────────────────────────────────────────────────────────────────
PAIR1    = "spy"   # data/raw/{pair}/1d.csv  (S&P 500 ETF)
PAIR2    = "qqq"   #                         (NASDAQ-100 ETF)
INTERVAL = "1d"

WINDOW           = 100   # EM 推定ウィンドウ幅 N (bars)
THRESHOLD        = 1.0   # エントリー z-score 閾値 c
N_EM_ITER        = 150   # 初回 EM 反復回数
REESTIMATE_EVERY = 20    # オンライン再推定頻度 (bars)

DATA_DIR   = _ROOT / "data"
OUTPUT_DIR = DATA_DIR / "signals" / "PairsTradingQFin05" / "standard" / "kalman_em"
# ──────────────────────────────────────────────────────────────────────────


def load_prices(pair: str, interval: str) -> pd.Series:
    path = DATA_DIR / "raw" / pair / f"{interval}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["close"].dropna().astype(float)


def compute_trades(
    signals_df: pd.DataFrame,
    pair1: str,
    pair2: str,
    hedge_ratio: float,
) -> pd.DataFrame:
    """Convert the signal column into entry/exit trade records.

    Long  spread → BUY pair1 / SELL pair2
    Short spread → SELL pair1 / BUY pair2
    """
    trades: list[dict] = []
    current = 0  # current position: +1 / 0 / -1

    for _, row in signals_df.iterrows():
        new = int(row["signal"])
        if new == current:
            continue

        # Close existing position
        if current != 0:
            trades.append({
                "datetime":        row["datetime"],
                "action":          "CLOSE",
                "direction":       "LONG" if current > 0 else "SHORT",
                f"{pair1}_side":   "SELL" if current > 0 else "BUY",
                f"{pair2}_side":   "BUY"  if current > 0 else "SELL",
                f"{pair1}_lots":   1.0,
                f"{pair2}_lots":   round(abs(hedge_ratio), 4),
                "z_score":         row["z_score"],
                "spread":          row["spread"],
            })

        # Open new position
        if new != 0:
            trades.append({
                "datetime":        row["datetime"],
                "action":          "OPEN",
                "direction":       "LONG" if new > 0 else "SHORT",
                f"{pair1}_side":   "BUY"  if new > 0 else "SELL",
                f"{pair2}_side":   "SELL" if new > 0 else "BUY",
                f"{pair1}_lots":   1.0,
                f"{pair2}_lots":   round(abs(hedge_ratio), 4),
                "z_score":         row["z_score"],
                "spread":          row["spread"],
            })

        current = new

    return pd.DataFrame(trades)


def main() -> None:
    print(f"[LOAD] {PAIR1} / {PAIR2}  interval={INTERVAL}")
    p1 = load_prices(PAIR1, INTERVAL)
    p2 = load_prices(PAIR2, INTERVAL)

    # Align on common dates
    common = p1.index.intersection(p2.index).sort_values()
    p1, p2 = p1.loc[common], p2.loc[common]
    print(f"[DATA] {len(common)} bars  {common[0].date()} ~ {common[-1].date()}")

    # Build log spread
    spread, hedge_ratio = build_spread(p1, p2)
    print(f"[SPREAD] hedge ratio β = {hedge_ratio:.4f}")

    # Generate signals
    signals_df, final_params = generate_signals(
        spread,
        dates=common,
        window=WINDOW,
        threshold=THRESHOLD,
        n_em_iter=N_EM_ITER,
        reestimate_every=REESTIMATE_EVERY,
    )

    long_n  = (signals_df.signal ==  1).sum()
    short_n = (signals_df.signal == -1).sum()
    flat_n  = (signals_df.signal ==  0).sum()
    print(f"[SIGNAL] {len(signals_df)} bars  long={long_n}  short={short_n}  flat={flat_n}")
    print(f"[EM final] {final_params}")

    if not final_params.is_valid():
        print("[WARN] Final params invalid — interpret results with caution.")

    # Convert signals to trade events
    trades_df = compute_trades(signals_df, PAIR1, PAIR2, hedge_ratio)
    print(f"[TRADES] {len(trades_df)} events  "
          f"open={(trades_df.action=='OPEN').sum()}  "
          f"close={(trades_df.action=='CLOSE').sum()}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{PAIR1}_{PAIR2}_{INTERVAL}"

    sig_path    = OUTPUT_DIR / f"{stem}_signals.csv"
    trades_path = OUTPUT_DIR / f"{stem}_trades.csv"

    signals_df.to_csv(sig_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print(f"[SAVE] {sig_path}")
    print(f"[SAVE] {trades_path}")

    # Preview first few trades
    if not trades_df.empty:
        print("\n--- trade preview (first 10) ---")
        print(trades_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
