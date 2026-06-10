"""Metrics for predicted-Sharpe pair selection."""
import pandas as pd


def sharpe_selection_metrics(predictions: pd.DataFrame, selected: pd.DataFrame) -> dict:
    """Evaluate predicted-Sharpe ranking against realized Sharpe labels."""
    if predictions.empty:
        return empty_sharpe_selection_metrics()
    ranked = predictions.sort_values("predicted_sharpe", ascending=False)
    return {
        "candidate_count": int(len(predictions)),
        "selected_count": int(len(selected)),
        "spearman_predicted_realized_sharpe": spearman_rank_correlation(ranked["predicted_sharpe"], ranked["realized_sharpe"]),
        "mean_realized_sharpe": float(predictions["realized_sharpe"].mean()),
        "top_prediction_mean_realized_sharpe": mean_top_prediction_realized_sharpe(predictions),
        "selected_mean_realized_sharpe": float(selected["realized_sharpe"].mean()) if not selected.empty else 0.0,
        "selected_mean_predicted_sharpe": float(selected["predicted_sharpe"].mean()) if not selected.empty else 0.0,
        "selected_mean_trade_count": float(selected["realized_trade_count"].mean()) if not selected.empty else 0.0,
    }


def mean_top_prediction_realized_sharpe(predictions: pd.DataFrame) -> float:
    """Return realized Sharpe for the monthly top predicted candidates."""
    rows = []
    for _, group in predictions.groupby("label_month", sort=True):
        rows.append(group.sort_values("predicted_sharpe", ascending=False).head(1))
    if not rows:
        return 0.0
    return float(pd.concat(rows, axis=0)["realized_sharpe"].mean())


def spearman_rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman rank correlation with pandas ranks."""
    if len(left) < 2:
        return 0.0
    value = left.rank().corr(right.rank())
    return float(value) if pd.notna(value) else 0.0


def empty_sharpe_selection_metrics() -> dict:
    """Return zeroed Sharpe selection metrics."""
    return {
        "candidate_count": 0,
        "selected_count": 0,
        "spearman_predicted_realized_sharpe": 0.0,
        "mean_realized_sharpe": 0.0,
        "top_prediction_mean_realized_sharpe": 0.0,
        "selected_mean_realized_sharpe": 0.0,
        "selected_mean_predicted_sharpe": 0.0,
        "selected_mean_trade_count": 0.0,
    }
