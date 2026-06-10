"""Correlation recovery labels."""
import numpy as np
import pandas as pd

from common.config import TradeConfig, WindowConfig
from common.models import CandidatePair, PairLabels


def label_candidate(
    candidate: CandidatePair,
    close: pd.DataFrame,
    returns: pd.DataFrame,
    windows: WindowConfig,
) -> PairLabels:
    """Build the future correlation-recovery label for one candidate."""
    position = int(returns.index.get_loc(candidate.date))
    y_corr, best_horizon = max_future_correlation_recovery(candidate, returns, position, windows)
    return PairLabels(
        y_corr=y_corr,
        best_horizon=best_horizon,
    )


def max_future_correlation_recovery(
    candidate: CandidatePair,
    returns: pd.DataFrame,
    position: int,
    windows: WindowConfig,
) -> tuple[float, int]:
    """Return max_h rho_future(t+h) - rho_now(t) for h in the configured range."""
    best_recovery = -np.inf
    best_horizon = windows.future_min_horizon
    for horizon in range(windows.future_min_horizon, windows.future_max_horizon + 1):
        start = position + horizon
        end = start + windows.future_corr_window
        future_window = returns[[candidate.asset_i, candidate.asset_j]].iloc[start:end]
        if len(future_window) < windows.future_corr_window:
            continue
        rho_future = float(future_window[candidate.asset_i].corr(future_window[candidate.asset_j]))
        recovery = rho_future - candidate.rho_now
        if recovery > best_recovery:
            best_recovery = recovery
            best_horizon = horizon
    if not np.isfinite(best_recovery):
        best_recovery = 0.0
    return float(best_recovery), int(best_horizon)


def simple_future_trade_outcome(
    candidate: CandidatePair,
    close: pd.DataFrame,
    returns_index: pd.Index,
    position: int,
    horizon: int,
    trade_config: TradeConfig,
) -> tuple[float, float]:
    """Simulate a single fixed-horizon spread trade after the decision date."""
    if abs(candidate.spread_z) < trade_config.z_entry:
        return 0.0, 0.0

    start_date = returns_index[position]
    end_position = min(position + horizon, len(returns_index) - 1)
    end_date = returns_index[end_position]
    path = close[[candidate.asset_i, candidate.asset_j]].loc[start_date:end_date].astype(float)
    if len(path) < 2:
        return 0.0, 0.0

    direction = -1.0 if candidate.spread_z > 0.0 else 1.0
    daily_returns = pair_trade_daily_returns(path, candidate.asset_i, candidate.asset_j, candidate.beta, direction)
    net_daily_returns = apply_round_trip_cost(daily_returns, trade_config.transaction_cost_bps)
    pnl = float(net_daily_returns.sum())
    risk = max_drawdown_from_returns(net_daily_returns)
    return pnl, risk


def pair_trade_daily_returns(
    prices: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    beta: float,
    direction: float,
) -> pd.Series:
    """Return daily long-spread or short-spread returns."""
    simple_returns = prices.pct_change().dropna()
    leg_return = direction * simple_returns[asset_i] - direction * beta * simple_returns[asset_j]
    denominator = 1.0 + abs(beta)
    return leg_return.divide(denominator).rename("pair_return")


def apply_round_trip_cost(daily_returns: pd.Series, transaction_cost_bps: float) -> pd.Series:
    """Apply open/close transaction costs to a daily return path."""
    if daily_returns.empty:
        return daily_returns
    adjusted = daily_returns.copy()
    cost = transaction_cost_bps / 10_000.0
    adjusted.iloc[0] = adjusted.iloc[0] - cost
    adjusted.iloc[-1] = adjusted.iloc[-1] - cost
    return adjusted


def max_drawdown_from_returns(daily_returns: pd.Series) -> float:
    """Return positive maximum drawdown from a daily return path."""
    equity = daily_returns.add(1.0).cumprod()
    drawdown = equity.divide(equity.cummax()).subtract(1.0)
    return float(abs(drawdown.min())) if not drawdown.empty else 0.0


def trade_direction_label(spread_z: float, pnl: float, z_entry: float) -> int:
    """Return 0 no-trade, 1 long i/short j, or 2 short i/long j."""
    if pnl <= 0.0 or abs(spread_z) < z_entry:
        return 0
    if spread_z < 0.0:
        return 1
    return 2


def position_size_label(spread_z: float, pnl: float, z_entry: float) -> float:
    """Return a bounded position-size target."""
    if pnl <= 0.0 or abs(spread_z) < z_entry:
        return 0.0
    return float(min(abs(spread_z) / max(3.0 * z_entry, 1e-12), 1.0))
