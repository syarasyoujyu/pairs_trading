"""Simulate ISEPT-selected pairs."""
import pandas as pd

from common.config import ImageConfig, TradingConfig
from images.calendar import future_months_window
from labels.pair_labels import history_window, pair_price_frame
from trading.strategy import simulate_pair_trade


def simulate_selected_pairs(
    selected_pairs: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    image_config: ImageConfig,
    trading_config: TradingConfig,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Simulate selected pairs and return portfolio returns, signals, and trades."""
    if selected_pairs.empty:
        return pd.Series(dtype=float, name="isept_return"), pd.DataFrame(), pd.DataFrame()
    index = next(iter(panel.values())).index
    return_series: list[pd.Series] = []
    signal_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for event_number, (_, row) in enumerate(selected_pairs.iterrows(), start=1):
        month_end = pd.Timestamp(row["month_end"])
        asset_i = str(row["asset_i"])
        asset_j = str(row["asset_j"])
        history_index = history_window(index, month_end, image_config.lookback_bars)
        trade_index = future_months_window(index, month_end, trading_config.trading_horizon_months)
        history_prices = pair_price_frame(panel, asset_i, asset_j, history_index)
        trading_prices = pair_price_frame(panel, asset_i, asset_j, trade_index)
        returns, signals, trades = simulate_pair_trade(history_prices, trading_prices, asset_i, asset_j, trading_config)
        event_id = event_identifier(event_number, month_end, asset_i, asset_j)
        return_series.append(returns.rename(event_id))
        signal_frames.append(enrich_signals(signals, row, event_id))
        trade_frames.append(enrich_trades(trades, row, event_id))
    return portfolio_returns(return_series), concat_frames(signal_frames), concat_frames(trade_frames)


def enrich_signals(signals: pd.DataFrame, selected_row: pd.Series, event_id: str) -> pd.DataFrame:
    """Add selected-pair metadata to signal rows."""
    frame = signals.copy()
    frame.insert(0, "event_id", event_id)
    frame.insert(1, "month_end", selected_row["month_end"])
    frame["predicted_sharpe"] = float(selected_row["predicted_sharpe"])
    frame["selection_rank"] = int(selected_row["rank"])
    return frame


def enrich_trades(trades: pd.DataFrame, selected_row: pd.Series, event_id: str) -> pd.DataFrame:
    """Add selected-pair metadata to trade rows."""
    frame = trades.copy()
    if frame.empty:
        return frame
    frame.insert(0, "event_id", event_id)
    frame.insert(1, "month_end", selected_row["month_end"])
    frame["predicted_sharpe"] = float(selected_row["predicted_sharpe"])
    frame["selection_rank"] = int(selected_row["rank"])
    return frame


def portfolio_returns(return_series: list[pd.Series]) -> pd.Series:
    """Average active selected-pair returns."""
    if not return_series:
        return pd.Series(dtype=float, name="isept_return")
    frame = pd.concat(return_series, axis=1).fillna(0.0)
    return frame.mean(axis=1).rename("isept_return")


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate non-empty frames."""
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, axis=0, ignore_index=True)


def event_identifier(event_number: int, month_end: pd.Timestamp, asset_i: str, asset_j: str) -> str:
    """Return stable selected-pair event id."""
    return f"event_{event_number:04d}_{month_end.strftime('%Y%m%d')}_{asset_i}_{asset_j}"
