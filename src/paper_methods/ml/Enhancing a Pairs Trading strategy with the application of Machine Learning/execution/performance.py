"""Performance calculations for pair and portfolio comparisons."""
import numpy as np
import pandas as pd


def pair_daily_returns(
    signals_df: pd.DataFrame,
    price_y: pd.Series,
    price_x: pd.Series,
) -> pd.Series:
    """Compute daily market-neutral returns for one pair."""
    signal = signals_df.set_index("datetime")["signal"].astype(float)
    signal.index = pd.to_datetime(signal.index).tz_localize(None)
    common = price_y.index.intersection(price_x.index).intersection(signal.index).sort_values()
    returns_y = price_y.reindex(common).pct_change()
    returns_x = price_x.reindex(common).pct_change()
    spread_return = 0.5 * returns_y - 0.5 * returns_x
    return (signal.reindex(common).shift(1).fillna(0.0) * spread_return).dropna()


def pair_trade_returns(
    signals_df: pd.DataFrame,
    price_y: pd.Series,
    price_x: pd.Series,
) -> pd.Series:
    """Compute one compounded return per non-flat holding episode."""
    signal = signals_df.set_index("datetime")["signal"].astype(float)
    signal.index = pd.to_datetime(signal.index).tz_localize(None)
    common = price_y.index.intersection(price_x.index).intersection(signal.index).sort_values()
    returns_y = price_y.reindex(common).pct_change()
    returns_x = price_x.reindex(common).pct_change()
    spread_return = 0.5 * returns_y - 0.5 * returns_x
    held_position = signal.reindex(common).shift(1).fillna(0.0)
    daily_returns = (held_position * spread_return).dropna()
    held_position = held_position.reindex(daily_returns.index).fillna(0.0)

    trade_ids = (held_position != held_position.shift(1)).cumsum()
    rows: list[float] = []
    for _, group_returns in daily_returns.groupby(trade_ids):
        position = float(held_position.reindex(group_returns.index).iloc[0])
        if position == 0.0:
            continue
        rows.append(total_return(group_returns))
    return pd.Series(rows, name="trade_return", dtype=float)


def portfolio_daily_returns(pair_returns: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight all pair return streams."""
    if not pair_returns:
        return pd.Series(dtype=float, name="portfolio")
    frame = pd.concat(pair_returns.values(), axis=1).fillna(0.0)
    frame.columns = list(pair_returns.keys())
    return frame.mean(axis=1).rename("portfolio")


def benchmark_daily_returns(price: pd.Series) -> pd.Series:
    """Buy-and-hold benchmark daily returns."""
    return price.pct_change().dropna().rename("benchmark")


def summarize_returns(name: str, daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Summarize a daily-return stream."""
    if daily_returns.empty:
        return {
            "model": name,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "active_days": 0,
            "total_days": 0,
        }
    return {
        "model": name,
        "total_return_pct": round(total_return(daily_returns) * 100, 4),
        "annualized_return_pct": round(annualized_return(daily_returns, periods_per_year) * 100, 4),
        "sharpe_ratio": round(sharpe_ratio(daily_returns, periods_per_year), 4),
        "max_drawdown_pct": round(max_drawdown(daily_returns) * 100, 4),
        "active_days": int((daily_returns != 0.0).sum()),
        "total_days": int(len(daily_returns)),
    }


def summarize_paper_trading_metrics(
    name: str,
    daily_returns: pd.Series,
    pair_returns: dict[str, pd.Series] | None = None,
    trade_returns: dict[str, pd.Series] | None = None,
    periods_per_year: int = 252,
) -> dict:
    """Summarize the trading metrics reported in Sarmento and Horta (2020)."""
    pair_returns = pair_returns or {}
    trade_returns = trade_returns or {}
    pair_total_returns = {
        pair: total_return(returns)
        for pair, returns in pair_returns.items()
        if not returns.empty
    }
    trade_frame = pd.concat(trade_returns.values(), ignore_index=True) if trade_returns else pd.Series(dtype=float)
    profitable_pairs = sum(value > 0.0 for value in pair_total_returns.values())
    total_pairs = len(pair_total_returns)
    profitable_trades = int((trade_frame > 0.0).sum()) if not trade_frame.empty else 0
    total_trades = int(len(trade_frame))
    return {
        "name": name,
        "SR": round(sharpe_ratio(daily_returns, periods_per_year), 4),
        "ROI_pct": round(total_return(daily_returns) * 100, 4) if not daily_returns.empty else 0.0,
        "MDD_pct": round(abs(max_drawdown(daily_returns)) * 100, 4) if not daily_returns.empty else 0.0,
        "portfolio_volatility_pct": round(float(daily_returns.std(ddof=1)) * 100, 4) if len(daily_returns) > 1 else 0.0,
        "annualization_factor": round(float(np.sqrt(periods_per_year)), 4),
        "days_of_portfolio_decline": int((daily_returns < 0.0).sum()) if not daily_returns.empty else 0,
        "total_pairs": total_pairs,
        "profitable_pairs": int(profitable_pairs),
        "profitable_pairs_pct": round(profitable_pairs / total_pairs * 100, 4) if total_pairs else 0.0,
        "total_trades": total_trades,
        "profitable_trades": profitable_trades,
        "unprofitable_trades": total_trades - profitable_trades,
    }


def total_return(daily_returns: pd.Series) -> float:
    """Compounded total return."""
    return float((1.0 + daily_returns).prod() - 1.0)


def annualized_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized compounded return."""
    years = len(daily_returns) / periods_per_year
    if years <= 0.0:
        return 0.0
    return float((1.0 + total_return(daily_returns)) ** (1.0 / years) - 1.0)


def sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe with zero risk-free rate."""
    std = float(daily_returns.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(daily_returns.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum drawdown of compounded returns."""
    cumulative = (1.0 + daily_returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    return float(drawdown.min())
