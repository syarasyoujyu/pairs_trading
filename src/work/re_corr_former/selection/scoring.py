"""Score and select ReCorrFormer candidate pairs."""
import pandas as pd

from common.config import ScoringConfig


def add_final_score(predictions: pd.DataFrame, config: ScoringConfig) -> pd.DataFrame:
    """Add final pair selection score from predicted correlation recovery only."""
    frame = predictions.copy()
    frame["final_score"] = frame["pred_corr"]
    return frame


def select_pairs_by_date(predictions: pd.DataFrame, config: ScoringConfig) -> pd.DataFrame:
    """Select Top-K pairs per date with optional no-overlap matching."""
    scored = add_final_score(predictions, config)
    selected_rows: list[pd.Series] = []
    for _, group in scored.groupby("date", sort=True):
        selected_rows.extend(select_one_date(group, config))
    if not selected_rows:
        return scored.head(0)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def select_one_date(group: pd.DataFrame, config: ScoringConfig) -> list[pd.Series]:
    """Select one date's pairs by greedy score ordering."""
    used_assets: set[str] = set()
    selected: list[pd.Series] = []
    ranked = group.sort_values("final_score", ascending=False)
    for _, row in ranked.iterrows():
        asset_i = str(row["asset_i"])
        asset_j = str(row["asset_j"])
        if not config.allow_asset_reuse and (asset_i in used_assets or asset_j in used_assets):
            continue
        selected.append(row)
        used_assets.update([asset_i, asset_j])
        if len(selected) >= config.top_k:
            break
    return selected
