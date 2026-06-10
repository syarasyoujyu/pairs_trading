"""Evaluation metrics for pair selection and trading."""
import numpy as np
import pandas as pd


def pair_selection_metrics(predictions: pd.DataFrame, selected: pd.DataFrame, top_k: int) -> dict:
    """Evaluate pair ranking quality on the test candidate set."""
    ranked = predictions.sort_values("final_score", ascending=False)
    top = ranked.head(top_k)
    return {
        "candidate_count": int(len(predictions)),
        "selected_count": int(len(selected)),
        "precision_at_k": precision_at_k(top),
        "ndcg_at_k": ndcg_at_k(ranked, top_k),
        "spearman_score_y_corr": spearman_rank_correlation(ranked["final_score"], ranked["y_corr"]),
        "top_k_mean_y_corr": float(top["y_corr"].mean()) if not top.empty else 0.0,
        "selected_mean_y_corr": float(selected["y_corr"].mean()) if not selected.empty else 0.0,
        "selected_mean_gap": float(selected["gap"].mean()) if not selected.empty else 0.0,
        "structural_break_rate_selected": structural_break_rate(selected),
    }


def precision_at_k(top: pd.DataFrame) -> float:
    """Return share of top candidates with high future correlation recovery."""
    if top.empty:
        return 0.0
    if "is_high_y_corr" in top.columns:
        return float(top["is_high_y_corr"].astype(bool).mean())
    return float((top["y_corr"] > 0.0).mean())


def ndcg_at_k(ranked: pd.DataFrame, top_k: int) -> float:
    """Compute NDCG@K using high-y_corr labels as relevance when available."""
    if ranked.empty:
        return 0.0
    relevance_values = relevance_for_ranking(ranked)
    relevance = relevance_values.head(top_k).to_numpy(dtype=float)
    ideal = np.sort(relevance_values.to_numpy(dtype=float))[::-1][:top_k]
    ideal_score = dcg(ideal)
    if ideal_score == 0.0:
        return 0.0
    return float(dcg(relevance) / ideal_score)


def relevance_for_ranking(frame: pd.DataFrame) -> pd.Series:
    """Return binary high-y_corr relevance when available."""
    if "is_high_y_corr" in frame.columns:
        return frame["is_high_y_corr"].astype(float)
    return frame["y_corr"].clip(lower=0.0)


def dcg(relevance: np.ndarray) -> float:
    """Return discounted cumulative gain."""
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance / discounts))


def spearman_rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman rank correlation without scipy."""
    if len(left) < 2:
        return 0.0
    return float(left.rank().corr(right.rank()))


def structural_break_rate(selected: pd.DataFrame) -> float:
    """Return share of selected pairs with negative future recovery."""
    if selected.empty:
        return 0.0
    return float((selected["y_corr"] < 0.0).mean())


def trading_metrics(portfolio_returns: pd.Series, trades: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """Compute ISEPT-compatible trading metrics for simulated returns."""
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
    """Return a zeroed trading metric row."""
    return {
        "total_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "volatility": 0.0,
        "hit_ratio": 0.0,
        "trade_count": 0,
        "avg_holding_days": 0.0,
    }


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio with zero risk-free rate."""
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    volatility = float(clean.std(ddof=1))
    if volatility <= 0.0 or not np.isfinite(volatility):
        return 0.0
    return float(clean.mean() / volatility * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Return positive maximum drawdown."""
    equity = returns.add(1.0).cumprod()
    drawdown = equity.divide(equity.cummax()).subtract(1.0)
    return float(abs(drawdown.min())) if not drawdown.empty else 0.0
