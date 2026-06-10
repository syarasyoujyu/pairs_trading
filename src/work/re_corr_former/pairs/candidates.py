"""Generate relationship-gap candidate pairs."""
from itertools import combinations

import numpy as np
import pandas as pd

from common.config import CandidateConfig, WindowConfig
from common.models import CandidatePair
from pairs.spread_features import hedge_beta, spread_series, spread_volatility, spread_zscore


def generate_candidate_pairs(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    windows: WindowConfig,
    config: CandidateConfig,
) -> list[CandidatePair]:
    """Generate candidates whose long correlation is high and current correlation has fallen."""
    returns = close.astype(float).pct_change().dropna()
    close = close.reindex(returns.index)
    volume = volume.reindex(returns.index)
    dollar_volume = close.astype(float) * volume.astype(float)
    positions = sample_decision_positions(len(returns), windows)
    candidates: list[CandidatePair] = []

    for position in positions:
        date_candidates = candidate_pairs_at_position(close, returns, dollar_volume, position, windows, config)
        candidates.extend(date_candidates)
    return candidates


def sample_decision_positions(n_rows: int, windows: WindowConfig) -> list[int]:
    """Return decision positions with enough past and future observations."""
    validate_lag_windows(windows)
    first = max(windows.lookback - 1, windows.long_corr_window, windows.short_corr_window)
    last = n_rows - windows.future_max_horizon - windows.future_corr_window
    if last <= first:
        raise ValueError("Not enough rows for the requested ReCorrFormer windows.")
    return list(range(first, last, windows.sample_stride))


def candidate_pairs_at_position(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    position: int,
    windows: WindowConfig,
    config: CandidateConfig,
) -> list[CandidatePair]:
    """Generate and rank relationship-gap candidates at one decision date."""
    long_returns = lagged_returns_window(returns, position, windows.long_corr_min_lag, windows.long_corr_window)
    short_returns = lagged_returns_window(returns, position, windows.short_corr_min_lag, windows.short_corr_window)
    rho_long = long_returns.corr()
    rho_now = short_returns.corr()
    liquid_assets = liquid_symbols_at_position(dollar_volume, position, config.min_liquidity_quantile)
    pair_rows = long_correlation_rank_rows(rho_long, rho_now, liquid_assets)
    rows: list[CandidatePair] = []

    for pair_row in pair_rows:
        if not is_relationship_gap_candidate(pair_row, config):
            continue
        rows.append(
            build_candidate_pair(
                close,
                returns.index[position],
                long_returns,
                short_returns,
                position,
                windows,
                pair_row["asset_i"],
                pair_row["asset_j"],
                pair_row["rho_long"],
                pair_row["rho_now"],
                pair_row["gap"],
                pair_row["long_corr_rank"],
                pair_row["long_corr_rank_fraction"],
                pair_row["long_corr_pair_count"],
            )
        )

    rows.sort(key=lambda candidate: (candidate.long_corr_rank_fraction, -candidate.gap))
    return rows[:config.max_candidates_per_date]


def liquid_symbols_at_position(dollar_volume: pd.DataFrame, position: int, min_quantile: float) -> list[str]:
    """Return assets with enough rolling dollar volume at a decision date."""
    liquidity = dollar_volume.iloc[max(0, position - 19):position + 1].mean(axis=0)
    threshold = float(liquidity.quantile(min_quantile))
    return liquidity[liquidity >= threshold].index.astype(str).tolist()


def long_correlation_rank_rows(rho_long: pd.DataFrame, rho_now: pd.DataFrame, liquid_assets: list[str]) -> list[dict]:
    """Return finite liquid pairs ranked by long-horizon correlation."""
    rows: list[dict] = []
    for asset_i, asset_j in combinations(sorted(liquid_assets), 2):
        long_value = float(rho_long.loc[asset_i, asset_j])
        now_value = float(rho_now.loc[asset_i, asset_j])
        if not np.isfinite(long_value) or not np.isfinite(now_value):
            continue
        rows.append(
            {
                "asset_i": asset_i,
                "asset_j": asset_j,
                "rho_long": long_value,
                "rho_now": now_value,
                "gap": long_value - now_value,
            }
        )
    rows.sort(key=lambda row: row["rho_long"], reverse=True)
    total = max(1, len(rows))
    for rank, row in enumerate(rows, start=1):
        row["long_corr_rank"] = rank
        row["long_corr_rank_fraction"] = rank / total
        row["long_corr_pair_count"] = total
    return rows


def is_relationship_gap_candidate(pair_row: dict, config: CandidateConfig) -> bool:
    """Return whether one ranked pair passes the relationship-gap candidate rule."""
    top_fraction = min(max(config.long_corr_top_fraction, 0.0), 1.0)
    top_count = max(1, int(np.ceil(pair_row["long_corr_pair_count"] * top_fraction)))
    return (
        pair_row["long_corr_rank"] <= top_count
        and pair_row["rho_long"] >= config.min_long_corr
        and pair_row["gap"] > config.min_gap
    )


def validate_lag_windows(windows: WindowConfig) -> None:
    """Validate lag-window configuration for short and long correlations."""
    if windows.short_corr_min_lag < 1 or windows.long_corr_min_lag < 1:
        raise ValueError("Correlation lags must be at least one day.")
    if windows.short_corr_window < windows.short_corr_min_lag:
        raise ValueError("short_corr_window must be greater than or equal to short_corr_min_lag.")
    if windows.long_corr_window < windows.long_corr_min_lag:
        raise ValueError("long_corr_window must be greater than or equal to long_corr_min_lag.")


def lagged_returns_window(returns: pd.DataFrame, position: int, min_lag: int, max_lag: int) -> pd.DataFrame:
    """Return rows from t-max_lag through t-min_lag for a decision position t."""
    start = position - max_lag
    end = position - min_lag + 1
    if start < 0 or end <= start:
        raise ValueError("Not enough rows for the requested lagged correlation window.")
    return returns.iloc[start:end]


def build_candidate_pair(
    close: pd.DataFrame,
    date: pd.Timestamp,
    long_returns: pd.DataFrame,
    short_returns: pd.DataFrame,
    position: int,
    windows: WindowConfig,
    asset_i: str,
    asset_j: str,
    rho_long: float,
    rho_now: float,
    gap: float,
    long_corr_rank: int,
    long_corr_rank_fraction: float,
    long_corr_pair_count: int,
) -> CandidatePair:
    """Build a candidate pair with spread features."""
    long_start = position - windows.long_corr_window
    long_end = position - windows.long_corr_min_lag + 1
    log_prices = np.log(close[[asset_i, asset_j]].iloc[long_start:long_end])
    beta = hedge_beta(log_prices[asset_i], log_prices[asset_j])
    spread = spread_series(log_prices[asset_i], log_prices[asset_j], beta)
    return CandidatePair(
        date=pd.Timestamp(date),
        asset_i=asset_i,
        asset_j=asset_j,
        rho_long=rho_long,
        rho_now=rho_now,
        gap=gap,
        long_corr_rank=long_corr_rank,
        long_corr_rank_fraction=long_corr_rank_fraction,
        long_corr_pair_count=long_corr_pair_count,
        long_corr_start_date=pd.Timestamp(long_returns.index[0]),
        long_corr_end_date=pd.Timestamp(long_returns.index[-1]),
        short_corr_start_date=pd.Timestamp(short_returns.index[0]),
        short_corr_end_date=pd.Timestamp(short_returns.index[-1]),
        spread_z=spread_zscore(spread),
        spread_volatility=spread_volatility(spread),
        beta=beta,
    )
