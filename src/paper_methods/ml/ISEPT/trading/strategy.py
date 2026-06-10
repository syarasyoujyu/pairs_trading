"""Trading-rule dispatch for ISEPT."""
import pandas as pd

from common.config import TradingConfig
from trading.gatev import simulate_gatev_pair
from trading.vidyamurthy import simulate_vidyamurthy_pair


def simulate_pair_trade(
    history_prices: pd.DataFrame,
    trading_prices: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    config: TradingConfig,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Simulate one selected pair with the configured execution rule."""
    if config.trade_rule == "gatev":
        return simulate_gatev_pair(history_prices, trading_prices, asset_i, asset_j, config)
    if config.trade_rule == "vidyamurthy":
        return simulate_vidyamurthy_pair(history_prices, trading_prices, asset_i, asset_j, config)
    raise ValueError(f"Unsupported trade_rule: {config.trade_rule}")
