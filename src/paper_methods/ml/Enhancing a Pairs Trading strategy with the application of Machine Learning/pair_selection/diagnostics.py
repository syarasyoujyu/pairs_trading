"""Cointegration and mean-reversion diagnostics for pair selection."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from common.models import CointegrationTestResult, PairDiagnostics
from pair_selection.spreads import build_spread, fit_hedge_ratio


def adf_t_statistic(series: pd.Series, max_lag: int = 0) -> float:
    """Estimate the ADF t-statistic for the lagged-level coefficient."""
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < max_lag + 4:
        return float("nan")

    delta = np.diff(values)
    lagged_level = values[:-1]
    rows: list[list[float]] = []
    target: list[float] = []

    for t in range(max_lag, len(delta)):
        row = [1.0, lagged_level[t]]
        for lag in range(1, max_lag + 1):
            row.append(delta[t - lag])
        rows.append(row)
        target.append(delta[t])

    x = np.asarray(rows, dtype=float)
    y = np.asarray(target, dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    dof = max(len(y) - x.shape[1], 1)
    sigma2 = float(residuals @ residuals / dof)
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    denominator = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if denominator == 0.0:
        return float("nan")
    return float(beta[1] / denominator)


def engle_granger_test(
    left: pd.Series,
    right: pd.Series,
    left_name: str,
    right_name: str,
    adf_max_lag: int | None = 0,
    p_value_threshold: float = 0.05,
    t_stat_threshold: float = -3.34,
) -> CointegrationTestResult:
    """Run Engle-Granger in both directions and keep the lower t-statistic."""
    first = _engle_granger_one_direction(
        left,
        right,
        left_name,
        right_name,
        adf_max_lag,
        p_value_threshold,
        t_stat_threshold,
    )
    second = _engle_granger_one_direction(
        right,
        left,
        right_name,
        left_name,
        adf_max_lag,
        p_value_threshold,
        t_stat_threshold,
    )
    return first if first.t_stat <= second.t_stat else second


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int | None = None) -> float:
    """Estimate H from the slope of lagged-difference volatility."""
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < min_lag + 2:
        return float("nan")

    max_lag_used = max_lag if max_lag is not None else min(100, len(values) // 2)
    max_lag_used = min(max_lag_used, len(values) // 2)
    lags = np.arange(min_lag, max_lag_used + 1)
    tau = np.array([np.std(values[lag:] - values[:-lag], ddof=1) for lag in lags])
    valid = tau > 0.0
    if valid.sum() < 2:
        return float("nan")

    slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return float(slope)


def half_life_bars(series: pd.Series) -> float:
    """Estimate mean-reversion half-life in bars from an AR(1) regression."""
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 3:
        return float("nan")

    lagged = values[:-1]
    delta = np.diff(values)
    design = np.column_stack([np.ones_like(lagged), lagged])
    beta = np.linalg.lstsq(design, delta, rcond=None)[0]
    speed = float(beta[1])
    if speed >= 0.0:
        return float("inf")
    return float(-np.log(2.0) / speed)


def mean_crossings_per_year(series: pd.Series, bars_per_year: int = 252) -> float:
    """Count spread mean crossings annualized by bars_per_year."""
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return 0.0

    centered = values - values.mean()
    signs = np.sign(centered)
    nonzero = signs != 0.0
    if nonzero.sum() < 2:
        return 0.0

    signs = signs[nonzero]
    crossings = int(np.sum(signs[1:] * signs[:-1] < 0.0))
    years = len(values) / float(bars_per_year)
    if years <= 0.0:
        return 0.0
    return float(crossings / years)


def evaluate_pair_rules(
    cointegration: CointegrationTestResult,
    bars_per_year: int = 252,
    min_half_life_bars: float = 1.0,
    max_half_life_bars: float | None = None,
    min_crossings_year: float = 12.0,
    max_hurst: float = 0.5,
) -> PairDiagnostics:
    """Apply the paper's four pair-selection filters."""
    max_half_life = float(max_half_life_bars or bars_per_year)
    spread = cointegration.spread
    hurst = hurst_exponent(spread)
    half_life = half_life_bars(spread)
    crossings = mean_crossings_per_year(spread, bars_per_year)
    reasons: list[str] = []

    if not cointegration.is_cointegrated:
        reasons.append("not cointegrated")
    if not np.isfinite(hurst) or hurst >= max_hurst:
        reasons.append("hurst >= 0.5")
    if not np.isfinite(half_life) or half_life < min_half_life_bars or half_life > max_half_life:
        reasons.append("half-life outside range")
    if crossings < min_crossings_year:
        reasons.append("too few mean crossings")

    return PairDiagnostics(
        asset_y=cointegration.dependent,
        asset_x=cointegration.independent,
        hedge_ratio=cointegration.hedge_ratio,
        intercept=cointegration.intercept,
        adf_t_stat=cointegration.t_stat,
        adf_p_value=cointegration.p_value,
        hurst=hurst,
        half_life_bars=half_life,
        crossings_per_year=crossings,
        spread_mean=float(spread.mean()),
        spread_std=float(spread.std(ddof=1)),
        is_selected=not reasons,
        rejection_reason="; ".join(reasons),
    )


def _engle_granger_one_direction(
    dependent: pd.Series,
    independent: pd.Series,
    dependent_name: str,
    independent_name: str,
    adf_max_lag: int | None,
    p_value_threshold: float,
    t_stat_threshold: float,
) -> CointegrationTestResult:
    hedge_ratio, intercept = fit_hedge_ratio(dependent, independent)
    spread = build_spread(dependent, independent, hedge_ratio, intercept)
    t_stat, p_value, critical_value = _adf_with_statsmodels(spread, adf_max_lag)

    is_cointegrated = bool(p_value < p_value_threshold)

    return CointegrationTestResult(
        dependent=dependent_name,
        independent=independent_name,
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        t_stat=t_stat,
        p_value=p_value,
        critical_value_5pct=critical_value,
        is_cointegrated=is_cointegrated,
        spread=spread,
    )


def _adf_with_statsmodels(
    spread: pd.Series,
    adf_max_lag: int | None,
) -> tuple[float, float | None, float | None]:
    autolag = "AIC" if adf_max_lag is None else None
    result = adfuller(spread.dropna().to_numpy(dtype=float), maxlag=adf_max_lag, autolag=autolag)
    return float(result[0]), float(result[1]), float(result[4]["5%"])
