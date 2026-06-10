"""PCA, density clustering, and statistical filters for pair selection."""
import numpy as np
import pandas as pd

from common.models import PairCandidate, SelectedPair
from pair_selection.clustering import density_cluster_labels, pair_candidates_from_clusters
from pair_selection.diagnostics import engle_granger_test, evaluate_pair_rules
from pair_selection.features import normalized_returns, pca_features


def select_pairs(
    prices: pd.DataFrame,
    n_components: int = 5,
    min_samples: int = 2,
    cluster_method: str = "auto",
    bars_per_day: int = 1,
    bars_per_year: int = 252,
    min_crossings_year: float = 12.0,
    use_log_prices: bool = True,
    max_pairs_per_cluster: int | None = None,
) -> tuple[list[SelectedPair], pd.Series]:
    """Select tradable pairs with PCA, density clustering, and four filters."""
    candidates, labels = diagnose_pair_candidates(
        prices,
        n_components=n_components,
        min_samples=min_samples,
        cluster_method=cluster_method,
        bars_per_day=bars_per_day,
        bars_per_year=bars_per_year,
        min_crossings_year=min_crossings_year,
        use_log_prices=use_log_prices,
        max_pairs_per_cluster=max_pairs_per_cluster,
    )
    selected = [
        SelectedPair(
            asset_y=candidate.asset_y,
            asset_x=candidate.asset_x,
            hedge_ratio=candidate.hedge_ratio,
            intercept=candidate.intercept,
            diagnostics=candidate.diagnostics,
            spread=candidate.spread,
        )
        for candidate in candidates
        if candidate.diagnostics.is_selected
    ]
    selected.sort(key=lambda pair: pair.diagnostics.adf_t_stat)
    return selected, labels


def diagnose_pair_candidates(
    prices: pd.DataFrame,
    n_components: int = 5,
    min_samples: int = 2,
    cluster_method: str = "auto",
    bars_per_day: int = 1,
    bars_per_year: int = 252,
    min_crossings_year: float = 12.0,
    use_log_prices: bool = True,
    max_pairs_per_cluster: int | None = None,
) -> tuple[list[PairCandidate], pd.Series]:
    """Evaluate every same-cluster candidate against the paper's filters."""
    clean_prices = prices.astype(float).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    returns = normalized_returns(clean_prices)
    features = pca_features(returns, n_components=n_components)
    labels = density_cluster_labels(features, method=cluster_method, min_samples=min_samples)
    raw_candidates = pair_candidates_from_clusters(
        labels,
        features=features,
        max_pairs_per_cluster=max_pairs_per_cluster,
    )
    price_basis = np.log(clean_prices) if use_log_prices else clean_prices
    candidates: list[PairCandidate] = []

    for left, right in raw_candidates:
        cointegration = engle_granger_test(
            price_basis[left],
            price_basis[right],
            left,
            right,
        )
        diagnostics = evaluate_pair_rules(
            cointegration,
            bars_per_year=bars_per_year,
            min_half_life_bars=float(bars_per_day),
            max_half_life_bars=float(bars_per_year),
            min_crossings_year=min_crossings_year,
        )
        candidates.append(
            PairCandidate(
                asset_y=diagnostics.asset_y,
                asset_x=diagnostics.asset_x,
                hedge_ratio=diagnostics.hedge_ratio,
                intercept=diagnostics.intercept,
                diagnostics=diagnostics,
                spread=cointegration.spread,
            )
        )

    candidates.sort(key=lambda pair: pair.diagnostics.adf_t_stat)
    return candidates, labels
