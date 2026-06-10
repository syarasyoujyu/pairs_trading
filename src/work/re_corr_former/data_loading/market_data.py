"""Load local OHLCV files for ReCorrFormer."""
from pathlib import Path

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def load_ohlcv_panel(
    raw_dir: Path,
    interval: str,
    max_assets: int,
    min_history_years: float = 8.0,
    exclude_symbols: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load aligned OHLCV frames for the most liquid symbols."""
    frames = load_symbol_frames(raw_dir, interval, exclude_symbols or set())
    selected_symbols = select_liquid_symbols(frames, max_assets, min_history_years)
    selected = {symbol: frames[symbol] for symbol in selected_symbols}
    return align_symbol_frames(selected)


def load_close_and_volume(
    raw_dir: Path,
    interval: str,
    max_assets: int,
    min_history_years: float = 8.0,
    exclude_symbols: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load close and volume frames, filtered to the most liquid symbols."""
    panel = load_ohlcv_panel(raw_dir, interval, max_assets, min_history_years, exclude_symbols)
    return close_volume_from_panel(panel)


def load_symbol_frames(raw_dir: Path, interval: str, exclude_symbols: set[str]) -> dict[str, pd.DataFrame]:
    """Load every available OHLCV symbol frame."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol_dir in sorted(raw_dir.iterdir()):
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name.lower()
        if symbol in exclude_symbols:
            continue
        path = symbol_dir / f"{interval}.csv"
        if not path.exists():
            continue
        frame = load_one_symbol_frame(path)
        if not frame.empty:
            frames[symbol] = frame
    if not frames:
        raise ValueError(f"No OHLCV frames found in {raw_dir}.")
    return frames


def load_one_symbol_frame(csv_path: Path) -> pd.DataFrame:
    """Load one OHLCV CSV with a normalized Date index."""
    frame = pd.read_csv(csv_path)
    frame.columns = [column.lower() for column in frame.columns]
    if "date" not in frame.columns:
        raise ValueError(f"{csv_path} must contain a Date column.")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame = frame.loc[:, list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def select_liquid_symbols(frames: dict[str, pd.DataFrame], max_assets: int, min_history_years: float = 8.0) -> list[str]:
    """Select liquid symbols after requiring enough price history."""
    if max_assets < 2:
        raise ValueError("max_assets must be at least 2.")
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
    """Align symbol frames to a common complete OHLCV index."""
    common_index = common_ohlcv_index(frames)
    aligned = {}
    for symbol, frame in frames.items():
        aligned[symbol] = frame.reindex(common_index).loc[:, list(OHLCV_COLUMNS)].dropna()
    return aligned


def common_ohlcv_index(frames: dict[str, pd.DataFrame]) -> pd.Index:
    """Return the common OHLCV index across selected symbols."""
    indexes = [frame.index for frame in frames.values()]
    common = indexes[0]
    for index in indexes[1:]:
        common = common.intersection(index)
    if common.empty:
        raise ValueError("Selected symbols do not have overlapping OHLCV dates.")
    return common.sort_values()


def close_volume_from_panel(panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned close and volume frames from an OHLCV panel."""
    close = pd.DataFrame({symbol: frame["close"] for symbol, frame in panel.items()})
    volume = pd.DataFrame({symbol: frame["volume"] for symbol, frame in panel.items()})
    return align_price_volume_frames(close, volume)


def align_price_volume_frames(close: pd.DataFrame, volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align close and volume frames and remove rows without complete values."""
    close = close.replace([np.inf, -np.inf], np.nan)
    volume = volume.replace([np.inf, -np.inf], np.nan)
    common_index = close.index.intersection(volume.index)
    common_columns = close.columns.intersection(volume.columns)
    close = close.loc[common_index, common_columns]
    volume = volume.loc[common_index, common_columns]
    valid_rows = close.notna().all(axis=1) & volume.notna().all(axis=1)
    return close.loc[valid_rows], volume.loc[valid_rows]


def most_liquid_symbols(close: pd.DataFrame, volume: pd.DataFrame, max_assets: int) -> list[str]:
    """Select symbols by average dollar volume."""
    if max_assets < 2:
        raise ValueError("max_assets must be at least 2.")
    dollar_volume = close.astype(float) * volume.astype(float)
    liquidity = dollar_volume.mean(axis=0).sort_values(ascending=False)
    return liquidity.head(max_assets).index.astype(str).tolist()
