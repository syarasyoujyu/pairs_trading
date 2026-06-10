"""Cluster-level movement plots based on the clustering input returns."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_cluster_return_movement_plots(
    returns: pd.DataFrame,
    labels: pd.Series,
    output_dir: Path,
) -> list[dict]:
    """Save one normalized-return movement plot for each non-noise cluster."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for cluster_id in sorted(int(label) for label in labels.unique() if int(label) >= 0):
        members = _cluster_members_in_returns(labels, returns, cluster_id)
        if not members:
            continue
        output_path = output_dir / f"cluster_{cluster_id}_normalized_returns.png"
        _save_one_cluster_plot(cluster_id, returns[members], output_path)
        rows.append(
            {
                "cluster": cluster_id,
                "member_count": len(members),
                "members": " ".join(members),
                "plot_path": str(output_path),
            }
        )

    return rows


def _cluster_members_in_returns(labels: pd.Series, returns: pd.DataFrame, cluster_id: int) -> list[str]:
    members = [str(asset) for asset, label in labels.items() if int(label) == cluster_id]
    return [member for member in sorted(members) if member in returns.columns]


def _save_one_cluster_plot(cluster_id: int, cluster_returns: pd.DataFrame, output_path: Path) -> None:
    movement_pct = cluster_returns.astype(float).multiply(100.0)
    mean_return = movement_pct.mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    ax.plot(movement_pct.index, movement_pct, color="#94a3b8", linewidth=0.65, alpha=0.28)
    ax.plot(mean_return.index, mean_return, color="#111827", linewidth=1.6, label="cluster mean")
    ax.axhline(0.0, color="#1f2937", linewidth=0.8, alpha=0.55)
    ax.set_title(f"Cluster {cluster_id}: daily normalized returns")
    ax.set_ylabel("Daily relative move (%)")
    ax.set_xlabel("Date")
    ax.grid(True, linewidth=0.5, alpha=0.28)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
