"""Build realized Sharpe labels for ISEPT feedback training."""
from itertools import combinations

import pandas as pd

from common.config import ImageConfig, TradingConfig
from images.calendar import future_months_window
from trading.metrics import sharpe_ratio
from trading.strategy import simulate_pair_trade


def realized_pair_labels_by_month(
    close_panel: dict[str, pd.DataFrame],
    month_ends: list[pd.Timestamp],
    symbols_by_month: dict[str, list[str]],
    image_config: ImageConfig,
    trading_config: TradingConfig,
) -> pd.DataFrame:
    """Compute realized next-month Sharpe labels for all candidate pairs."""
    rows: list[dict] = []
    index = next(iter(close_panel.values())).index
    for month_end in month_ends:
        month_key = month_end.strftime("%Y-%m-%d")
        symbols = symbols_by_month[month_key]
        history_index = history_window(index, month_end, image_config.lookback_bars)
        trade_index = future_months_window(index, month_end, trading_config.trading_horizon_months)
        if trade_index.empty:
            continue
        for asset_i, asset_j in combinations(symbols, 2):
            history_prices = pair_price_frame(close_panel, asset_i, asset_j, history_index)
            trading_prices = pair_price_frame(close_panel, asset_i, asset_j, trade_index)
            returns, _, trades = simulate_pair_trade(history_prices, trading_prices, asset_i, asset_j, trading_config)
            rows.append(
                {
                    "label_month": month_key,
                    "asset_i": asset_i,
                    "asset_j": asset_j,
                    "trade_rule": trading_config.trade_rule,
                    "realized_sharpe": sharpe_ratio(returns),
                    "trade_count": int(len(trades)),
                    "total_return_pct": float((returns.add(1.0).prod() - 1.0) * 100.0) if not returns.empty else 0.0,
                }
            )
    return pd.DataFrame(rows)


def history_window(index: pd.Index, month_end: pd.Timestamp, lookback_bars: int) -> pd.Index:
    """Return the lookback index ending at month_end."""
    end_position = int(index.get_loc(month_end))
    start_position = max(0, end_position - lookback_bars + 1)
    return index[start_position:end_position + 1]


def pair_price_frame(panel: dict[str, pd.DataFrame], asset_i: str, asset_j: str, dates: pd.Index) -> pd.DataFrame:
    """Return pair close prices over dates."""
    return pd.DataFrame(
        {
            asset_i: panel[asset_i].reindex(dates)["close"],
            asset_j: panel[asset_j].reindex(dates)["close"],
        },
        index=dates,
    ).dropna()
