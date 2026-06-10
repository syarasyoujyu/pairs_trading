"""Performance metrics for ISEPT trading simulation."""
import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio with zero risk-free rate."""
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    volatility = float(clean.std(ddof=1))
    if volatility <= 0.0 or not np.isfinite(volatility):
        return 0.0
    return float(clean.mean() / volatility * np.sqrt(periods_per_year))


def trading_metrics(portfolio_returns: pd.Series, trades: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """Compute strategy-level trading metrics."""
    clean = portfolio_returns.dropna().astype(float)
    if clean.empty:
        return empty_trading_metrics()
    total_return = float(clean.add(1.0).prod() - 1.0)
    volatility = float(clean.std(ddof=1) * np.sqrt(periods_per_year))
    return {
        "total_return_pct": total_return * 100.0,
        "sharpe_ratio": sharpe_ratio(clean, periods_per_year),
        "max_drawdown_pct": max_drawdown(clean) * 100.0,
        "volatility": volatility,
        "hit_ratio": float((clean > 0.0).mean()),
        "trade_count": int(len(trades)),
        "avg_holding_days": float(trades["holding_days"].mean()) if not trades.empty else 0.0,
    }


def empty_trading_metrics() -> dict:
    """Return zero metrics for an empty strategy."""
    return {
        "total_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "volatility": 0.0,
        "hit_ratio": 0.0,
        "trade_count": 0,
        "avg_holding_days": 0.0,
    }


def max_drawdown(returns: pd.Series) -> float:
    """Return positive max drawdown."""
    equity = returns.add(1.0).cumprod()
    drawdown = equity.divide(equity.cummax()).subtract(1.0)
    return float(abs(drawdown.min())) if not drawdown.empty else 0.0
