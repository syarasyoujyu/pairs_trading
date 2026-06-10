"""Reporting helpers for relationship-gap candidate pairs."""
import numpy as np
import pandas as pd

from common.config import CandidateConfig
from common.models import CandidatePair


def candidate_pairs_to_frame(candidates: list[CandidatePair], config: CandidateConfig) -> pd.DataFrame:
    """Convert generated candidates into a readable DataFrame."""
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "date": candidate.date,
                "pair": f"{candidate.asset_i}/{candidate.asset_j}",
                "asset_i": candidate.asset_i,
                "asset_j": candidate.asset_j,
                "rho_long": candidate.rho_long,
                "rho_now": candidate.rho_now,
                "gap": candidate.gap,
                "long_corr_rank": candidate.long_corr_rank,
                "long_corr_rank_fraction": candidate.long_corr_rank_fraction,
                "long_corr_pair_count": candidate.long_corr_pair_count,
                "long_corr_start_date": candidate.long_corr_start_date,
                "long_corr_end_date": candidate.long_corr_end_date,
                "short_corr_start_date": candidate.short_corr_start_date,
                "short_corr_end_date": candidate.short_corr_end_date,
                "spread_z": candidate.spread_z,
                "spread_volatility": candidate.spread_volatility,
                "beta": candidate.beta,
                "min_long_corr": config.min_long_corr,
                "long_corr_top_fraction": config.long_corr_top_fraction,
                "min_gap": config.min_gap,
                "passes_long_corr": candidate.rho_long >= config.min_long_corr,
                "passes_long_corr_top_fraction": passes_long_corr_top_fraction(candidate, config),
                "passes_gap": candidate.gap > config.min_gap,
                "passed_liquidity_filter": True,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["date", "gap", "rho_long"], ascending=[True, False, False]).reset_index(drop=True)
    frame["rank_by_gap_on_date"] = frame.groupby("date").cumcount() + 1
    ordered_columns = [
        "date",
        "rank_by_gap_on_date",
        "pair",
        "asset_i",
        "asset_j",
        "rho_long",
        "rho_now",
        "gap",
        "long_corr_rank",
        "long_corr_rank_fraction",
        "long_corr_pair_count",
        "long_corr_start_date",
        "long_corr_end_date",
        "short_corr_start_date",
        "short_corr_end_date",
        "spread_z",
        "spread_volatility",
        "beta",
        "min_long_corr",
        "long_corr_top_fraction",
        "min_gap",
        "passes_long_corr",
        "passes_long_corr_top_fraction",
        "passes_gap",
        "passed_liquidity_filter",
    ]
    return frame[ordered_columns]


def passes_long_corr_top_fraction(candidate: CandidatePair, config: CandidateConfig) -> bool:
    """Return whether the candidate passes the configured long-correlation top fraction."""
    top_count = max(1, int(np.ceil(candidate.long_corr_pair_count * config.long_corr_top_fraction)))
    return candidate.long_corr_rank <= top_count


def summarize_candidate_pairs(candidate_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize relationship-gap candidate counts and top pairs by date."""
    if candidate_frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "candidate_count",
                "mean_rho_long",
                "mean_rho_now",
                "mean_gap",
                "max_gap",
                "top_pair",
                "top_pair_rho_long",
                "top_pair_rho_now",
                "top_pair_gap",
            ]
        )
    summary = (
        candidate_frame.groupby("date")
        .agg(
            candidate_count=("pair", "count"),
            mean_rho_long=("rho_long", "mean"),
            mean_rho_now=("rho_now", "mean"),
            mean_gap=("gap", "mean"),
            max_gap=("gap", "max"),
        )
        .reset_index()
    )
    top_pairs = (
        candidate_frame.sort_values(["date", "rank_by_gap_on_date"])
        .groupby("date", as_index=False)
        .first()[["date", "pair", "rho_long", "rho_now", "gap"]]
        .rename(
            columns={
                "pair": "top_pair",
                "rho_long": "top_pair_rho_long",
                "rho_now": "top_pair_rho_now",
                "gap": "top_pair_gap",
            }
        )
    )
    return summary.merge(top_pairs, on="date", how="left")
