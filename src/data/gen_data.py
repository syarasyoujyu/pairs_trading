"""Download price data for local experiments.

Default mode fetches the current SPY holdings from State Street and downloads
daily adjusted OHLCV data for every referenced equity plus SPY itself.
"""
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
import re

import pandas as pd
import requests
import yfinance as yf


SPY_HOLDINGS_URL = (
    "https://www.ssga.com/library-content/products/fund-data/"
    "etfs/us/holdings-daily-us-en-spy.xlsx"
)
INTERVAL = "1d"
PERIOD = "10y"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
UNIVERSE_DIR = Path(__file__).parent.parent.parent / "data" / "universe"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}


def download_spy_holdings(url: str = SPY_HOLDINGS_URL) -> pd.DataFrame:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    raw = pd.read_excel(BytesIO(response.content), sheet_name="holdings", header=None, engine="openpyxl")
    header_row = _find_header_row(raw, "Ticker")
    holdings = pd.read_excel(
        BytesIO(response.content),
        sheet_name="holdings",
        header=header_row,
        engine="openpyxl",
    )
    holdings = _filter_equity_holdings(holdings)
    holdings["yfinance_symbol"] = holdings["Ticker"].map(to_yfinance_symbol)
    holdings["storage_symbol"] = holdings["yfinance_symbol"].map(to_storage_symbol)
    return holdings.reset_index(drop=True)


def spy_holding_symbols(holdings: pd.DataFrame, include_spy: bool = True) -> list[str]:
    symbols = holdings["yfinance_symbol"].dropna().astype(str).drop_duplicates().tolist()
    if include_spy and "SPY" not in symbols:
        symbols.insert(0, "SPY")
    return symbols


def fetch_prices(
    symbols: list[str],
    interval: str,
    period: str,
    chunk_size: int = 40,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for start in range(0, len(symbols), chunk_size):
        chunk = symbols[start:start + chunk_size]
        print(f"[FETCH] {start + 1}-{start + len(chunk)} / {len(symbols)}")
        downloaded = yf.download(
            tickers=chunk,
            interval=interval,
            period=period,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        data.update(_split_downloaded_prices(downloaded, chunk))
    return data


def save_prices(data: dict[str, pd.DataFrame], output_dir: Path, interval: str) -> None:
    for symbol, frame in data.items():
        if frame.empty:
            print(f"[WARN] no data for {symbol}")
            continue
        symbol_dir = output_dir / to_storage_symbol(symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        path = symbol_dir / f"{interval}.csv"
        frame.to_csv(path)
        print(f"[SAVE] {path} rows={len(frame)}")


def save_spy_holdings(holdings: pd.DataFrame, universe_dir: Path) -> Path:
    universe_dir.mkdir(parents=True, exist_ok=True)
    path = universe_dir / "spy_holdings.csv"
    holdings.to_csv(path, index=False)
    print(f"[SAVE] {path} rows={len(holdings)}")
    return path


def to_yfinance_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def to_storage_symbol(symbol: str) -> str:
    return symbol.strip().lower().replace(".", "-")


def _find_header_row(raw: pd.DataFrame, column_name: str) -> int:
    matches = raw.index[raw.eq(column_name).any(axis=1)].tolist()
    if not matches:
        raise ValueError(f"Could not find holdings header row containing {column_name!r}.")
    return int(matches[0])


def _filter_equity_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    equities = holdings.dropna(subset=["Ticker"]).copy()
    equities["Ticker"] = equities["Ticker"].astype(str).str.strip()
    if "Local Currency" in equities.columns:
        equities = equities[equities["Local Currency"].astype(str).str.upper().eq("USD")]
    return equities[equities["Ticker"].map(_is_equity_ticker)].copy()


def _is_equity_ticker(symbol: str) -> bool:
    normalized = symbol.strip().upper()
    if not normalized or normalized == "NAN" or normalized in {"-", "--", "CASH"}:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9]*(\.[A-Z])?", normalized))


def _split_downloaded_prices(downloaded: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    if downloaded.empty:
        return data

    if isinstance(downloaded.columns, pd.MultiIndex):
        for symbol in symbols:
            if symbol not in downloaded.columns.get_level_values(0):
                data[symbol] = pd.DataFrame()
                continue
            frame = downloaded[symbol].dropna(how="all").copy()
            data[symbol] = _normalize_price_columns(frame)
        return data

    data[symbols[0]] = _normalize_price_columns(downloaded.dropna(how="all").copy())
    return data


def _normalize_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    return frame.dropna(subset=["close"])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--interval", default=INTERVAL)
    parser.add_argument("--period", default=PERIOD)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-spy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    holdings = download_spy_holdings()
    save_spy_holdings(holdings, UNIVERSE_DIR)
    symbols = spy_holding_symbols(holdings, include_spy=not args.no_spy)
    if args.limit is not None:
        symbols = symbols[:args.limit]
    print(f"[UNIVERSE] symbols={len(symbols)} interval={args.interval} period={args.period}")
    data = fetch_prices(symbols, args.interval, args.period, args.chunk_size)
    save_prices(data, OUTPUT_DIR, args.interval)


if __name__ == "__main__":
    main()
