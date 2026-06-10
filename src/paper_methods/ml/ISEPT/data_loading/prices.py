"""Load OHLCV data for ISEPT."""
from pathlib import Path

import pandas as pd


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def load_ohlcv_panel(
    raw_dir: Path,
    interval: str,
    max_assets: int,
    exclude_symbols: set[str],
    min_history_years: float = 8.0,
) -> dict[str, pd.DataFrame]:
    """Load aligned OHLCV frames for liquid symbols."""
    frames = load_symbol_frames(raw_dir, interval, exclude_symbols)
    selected_symbols = select_liquid_symbols(frames, max_assets, min_history_years)
    selected = {symbol: frames[symbol] for symbol in selected_symbols}
    return align_symbol_frames(selected)


def load_symbol_frames(raw_dir: Path, interval: str, exclude_symbols: set[str]) -> dict[str, pd.DataFrame]:
    """Load every symbol frame available under data/raw."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol_dir in sorted(raw_dir.iterdir()):
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name.lower()
        if symbol in exclude_symbols:
            continue
        csv_path = symbol_dir / f"{interval}.csv"
        if not csv_path.exists():
            continue
        frame = load_one_symbol_frame(csv_path)
        if not frame.empty:
            frames[symbol] = frame
    if not frames:
        raise ValueError(f"No OHLCV frames found in {raw_dir}.")
    return frames


def load_one_symbol_frame(csv_path: Path) -> pd.DataFrame:
    """Load one OHLCV CSV."""
    frame = pd.read_csv(csv_path)
    frame.columns = [column.lower() for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame = frame.loc[:, list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return frame.dropna()


def select_liquid_symbols(frames: dict[str, pd.DataFrame], max_assets: int, min_history_years: float = 8.0) -> list[str]:
    """Select liquid symbols after requiring enough price history."""
    rows = []
    for symbol, frame in frames.items():
        dollar_volume = frame["close"].astype(float) * frame["volume"].astype(float)
        rows.append(
            {
                "symbol": symbol,
                "mean_dollar_volume": float(dollar_volume.mean()),
                "history_years": history_years(frame.index),
            }
        )
    ranking = liquid_symbol_ranking(pd.DataFrame(rows), max_assets, min_history_years)
    return ranking["symbol"].head(max_assets).astype(str).tolist()


def liquid_symbol_ranking(ranking: pd.DataFrame, max_assets: int, min_history_years: float) -> pd.DataFrame:
    """Return a liquidity ranking with a history-length filter when possible."""
    ordered = ranking.sort_values("mean_dollar_volume", ascending=False).reset_index(drop=True)
    if min_history_years <= 0.0:
        return ordered
    eligible = ordered[ordered["history_years"] >= min_history_years].copy()
    if len(eligible) >= max_assets:
        return eligible.sort_values("mean_dollar_volume", ascending=False).reset_index(drop=True)
    if len(eligible) >= 2:
        return eligible.sort_values("mean_dollar_volume", ascending=False).reset_index(drop=True)
    return ordered


def history_years(index: pd.Index) -> float:
    """Return approximate calendar years covered by an index."""
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if dates.empty:
        return 0.0
    span_days = max(0, (dates.max() - dates.min()).days)
    return span_days / 365.25


def align_symbol_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align symbols to a common date index with complete OHLCV values."""
    common_index = common_price_index(frames)
    aligned = {}
    for symbol, frame in frames.items():
        aligned[symbol] = frame.reindex(common_index).dropna()
    return aligned


def common_price_index(frames: dict[str, pd.DataFrame]) -> pd.Index:
    """Return the common index across all selected symbols."""
    indexes = [frame.index for frame in frames.values()]
    common = indexes[0]
    for index in indexes[1:]:
        common = common.intersection(index)
    if common.empty:
        raise ValueError("Selected symbols do not have overlapping OHLCV dates.")
    return common.sort_values()


def close_frame(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a close-price frame from a symbol panel."""
    return pd.DataFrame({symbol: frame["close"] for symbol, frame in panel.items()}).dropna()
