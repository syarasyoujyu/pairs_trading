"""Data loading for local data/raw price files."""
from pathlib import Path

import pandas as pd


def load_price_frame(
    raw_dir: Path,
    interval: str,
    symbols: list[str] | None = None,
    drop_missing_rows: bool = True,
) -> pd.DataFrame:
    """Load close prices from data/raw/{symbol}/{interval}.csv."""
    raw_dir = Path(raw_dir)
    selected_symbols = symbols or sorted(path.name for path in raw_dir.iterdir() if path.is_dir())
    series: list[pd.Series] = []

    for symbol in selected_symbols:
        path = raw_dir / symbol.lower() / f"{interval}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if "close" not in frame.columns:
            raise ValueError(f"{path} does not contain a 'close' column.")
        series.append(frame["close"].dropna().astype(float).rename(symbol.lower()))

    prices = pd.concat(series, axis=1).sort_index()
    if drop_missing_rows:
        return prices.dropna(axis=0, how="any")
    return prices


def load_spy_universe_symbols(universe_path: Path, include_spy: bool = False) -> list[str]:
    """Load yfinance symbols saved by src/data/gen_data.py."""
    frame = pd.read_csv(universe_path)
    symbols = frame["yfinance_symbol"].dropna().astype(str).str.lower().drop_duplicates().tolist()
    if include_spy and "spy" not in symbols:
        symbols.insert(0, "spy")
    return symbols


def filter_price_history(
    prices: pd.DataFrame,
    min_non_null: int,
    max_missing_ratio: float = 0.02,
) -> pd.DataFrame:
    """Keep assets with enough observations and acceptable missingness."""
    non_null = prices.notna().sum(axis=0)
    missing_ratio = prices.isna().mean(axis=0)
    columns = non_null[(non_null >= min_non_null) & (missing_ratio <= max_missing_ratio)].index
    return prices.loc[:, columns].dropna(axis=0, how="any")
