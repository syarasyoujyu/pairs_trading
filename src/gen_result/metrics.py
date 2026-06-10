"""
Performance metrics for the pairs trading strategy.

Daily return model (market-neutral, equal dollar weighting):
    r_strategy_k = position_{k-1} * (0.5 * r_pair1_k - 0.5 * r_pair2_k)

where position ∈ {+1, 0, -1} comes from the signals CSV.
"""
import numpy as np
import pandas as pd


def compute_daily_returns(
    signals_df: pd.DataFrame,
    price1: pd.Series,
    price2: pd.Series,
) -> pd.Series:
    """Compute per-day strategy returns.

    Position at end of day k is taken from signals_df.
    That position is held from open of k+1, earning return r_{k+1}.

    Args:
        signals_df: Output of SignalGenerator — must have 'datetime' and 'signal' columns.
        price1:     Close prices of pair1 (e.g. SPY), DatetimeIndex.
        price2:     Close prices of pair2 (e.g. QQQ), DatetimeIndex.

    Returns:
        Daily return series aligned to price1/price2 dates.
    """
    # Daily simple returns for each leg
    r1 = price1.pct_change()
    r2 = price2.pct_change()

    # Build position series on the full price date range
    sig = signals_df.set_index("datetime")["signal"]
    sig.index = pd.to_datetime(sig.index)

    common = price1.index.intersection(price2.index)
    pos = sig.reindex(common, method="ffill").fillna(0)

    # Strategy return = yesterday's position * today's spread return
    # Long  spread (+1): long pair1, short pair2
    # Short spread (-1): short pair1, long pair2
    spread_ret = 0.5 * r1.reindex(common) - 0.5 * r2.reindex(common)
    daily_ret = pos.shift(1) * spread_ret

    return daily_ret.dropna().rename("strategy")


def compute_benchmark_returns(price: pd.Series) -> pd.Series:
    """Buy-and-hold daily returns for a single asset."""
    return price.pct_change().dropna().rename("benchmark")


def sharpe_ratio(daily_ret: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    if daily_ret.std() == 0:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(periods_per_year))


def total_return(daily_ret: pd.Series) -> float:
    """Compounded total return as a fraction (e.g. 0.25 = +25%)."""
    return float((1 + daily_ret).prod() - 1)


def annualised_return(daily_ret: pd.Series, periods_per_year: int = 252) -> float:
    n_years = len(daily_ret) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float((1 + total_return(daily_ret)) ** (1 / n_years) - 1)


def max_drawdown(daily_ret: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction."""
    cum = (1 + daily_ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def win_rate(daily_ret: pd.Series) -> float:
    """Fraction of days with positive return (among active days)."""
    active = daily_ret[daily_ret != 0]
    if len(active) == 0:
        return float("nan")
    return float((active > 0).mean())


def build_report(
    daily_ret: pd.Series,
    benchmark_ret: pd.Series | None = None,
) -> dict:
    report = {
        "total_return_%":      round(total_return(daily_ret) * 100, 2),
        "annualised_return_%": round(annualised_return(daily_ret) * 100, 2),
        "sharpe_ratio":        round(sharpe_ratio(daily_ret), 3),
        "max_drawdown_%":      round(max_drawdown(daily_ret) * 100, 2),
        "win_rate_%":          round(win_rate(daily_ret) * 100, 2),
        "active_days":         int((daily_ret != 0).sum()),
        "total_days":          int(len(daily_ret)),
    }
    if benchmark_ret is not None:
        report["benchmark_total_return_%"]      = round(total_return(benchmark_ret) * 100, 2)
        report["benchmark_annualised_return_%"] = round(annualised_return(benchmark_ret) * 100, 2)
        report["benchmark_sharpe_ratio"]        = round(sharpe_ratio(benchmark_ret), 3)
        report["benchmark_max_drawdown_%"]      = round(max_drawdown(benchmark_ret) * 100, 2)
    return report
