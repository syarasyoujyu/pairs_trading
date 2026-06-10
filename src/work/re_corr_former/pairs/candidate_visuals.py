"""Visual reports for relationship-gap candidate pair sets."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from common.config import WindowConfig


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def candidate_pair_set_frame(candidate_frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per selected candidate pair."""
    if candidate_frame.empty:
        return pd.DataFrame(
            columns=[
                "pair",
                "asset_i",
                "asset_j",
                "selection_count",
                "first_selected_date",
                "last_selected_date",
                "max_gap",
                "mean_gap",
                "max_rho_long",
                "min_rho_now",
                "mean_spread_z",
            ]
        )
    pair_set = (
        candidate_frame.groupby(["pair", "asset_i", "asset_j"], as_index=False)
        .agg(
            selection_count=("date", "count"),
            first_selected_date=("date", "min"),
            last_selected_date=("date", "max"),
            max_gap=("gap", "max"),
            mean_gap=("gap", "mean"),
            max_rho_long=("rho_long", "max"),
            min_rho_now=("rho_now", "min"),
            mean_spread_z=("spread_z", "mean"),
        )
        .sort_values(["selection_count", "max_gap", "max_rho_long"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    pair_set["pair_set_rank"] = pair_set.index + 1
    ordered_columns = [
        "pair_set_rank",
        "pair",
        "asset_i",
        "asset_j",
        "selection_count",
        "first_selected_date",
        "last_selected_date",
        "max_gap",
        "mean_gap",
        "max_rho_long",
        "min_rho_now",
        "mean_spread_z",
    ]
    return pair_set[ordered_columns]


def save_candidate_pair_movement_images(
    pair_set: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    close: pd.DataFrame,
    output_dir: Path,
    windows: WindowConfig,
    image_limit: int,
    table_step: int,
) -> pd.DataFrame:
    """Save normalized price movement images for selected candidate pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    selected_pair_set = limited_pair_set(pair_set, image_limit)
    for _, pair_row in selected_pair_set.iterrows():
        pair_candidates = candidate_frame[
            (candidate_frame["asset_i"] == pair_row["asset_i"])
            & (candidate_frame["asset_j"] == pair_row["asset_j"])
        ].copy()
        image_path = output_dir / pair_image_filename(pair_row)
        save_one_pair_movement_image(pair_row, pair_candidates, close, windows, table_step, image_path)
        manifest_rows.append(
            {
                "pair_set_rank": int(pair_row["pair_set_rank"]),
                "pair": pair_row["pair"],
                "asset_i": pair_row["asset_i"],
                "asset_j": pair_row["asset_j"],
                "selection_count": int(pair_row["selection_count"]),
                "max_gap": float(pair_row["max_gap"]),
                "image_path": str(image_path),
            }
        )
    return pd.DataFrame(manifest_rows)


def limited_pair_set(pair_set: pd.DataFrame, image_limit: int) -> pd.DataFrame:
    """Return the pair set rows that should receive images."""
    if image_limit <= 0:
        return pair_set
    return pair_set.head(image_limit)


def save_one_pair_movement_image(
    pair_row: pd.Series,
    pair_candidates: pd.DataFrame,
    close: pd.DataFrame,
    windows: WindowConfig,
    table_step: int,
    image_path: Path,
) -> None:
    """Save one pair's normalized movement chart and table-like heatmap."""
    asset_i = str(pair_row["asset_i"])
    asset_j = str(pair_row["asset_j"])
    price_path = pair_price_window(close, pair_candidates, asset_i, asset_j, windows)
    relative = price_path.divide(price_path.iloc[0]).subtract(1.0).multiply(100.0)
    transition_table = build_transition_table(relative, table_step)
    candidate_dates = pd.to_datetime(pair_candidates["date"]).sort_values()

    fig = plt.figure(figsize=figure_size(transition_table), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.35])
    price_axis = fig.add_subplot(grid[0])
    table_axis = fig.add_subplot(grid[1])

    plot_relative_price_panel(price_axis, relative, pair_row, candidate_dates)
    plot_transition_table_panel(table_axis, transition_table, candidate_dates)

    fig.savefig(image_path, dpi=160)
    plt.close(fig)


def pair_price_window(
    close: pd.DataFrame,
    pair_candidates: pd.DataFrame,
    asset_i: str,
    asset_j: str,
    windows: WindowConfig,
) -> pd.DataFrame:
    """Return the price window around this pair's selected dates."""
    candidate_dates = pd.to_datetime(pair_candidates["date"]).sort_values()
    first_position = int(close.index.get_loc(candidate_dates.iloc[0]))
    last_position = int(close.index.get_loc(candidate_dates.iloc[-1]))
    start_position = max(0, first_position - windows.long_corr_window)
    end_position = min(len(close.index) - 1, last_position + windows.future_max_horizon)
    return close[[asset_i, asset_j]].iloc[start_position:end_position + 1].astype(float).dropna()


def build_transition_table(relative: pd.DataFrame, table_step: int) -> pd.DataFrame:
    """Build a compact transition table from relative price movement."""
    step = max(1, table_step)
    positions = list(range(0, len(relative), step))
    if positions[-1] != len(relative) - 1:
        positions.append(len(relative) - 1)
    sampled = relative.iloc[positions]
    asset_i = relative.columns[0]
    asset_j = relative.columns[1]
    table = pd.DataFrame(
        {
            f"{asset_i} rel %": sampled[asset_i],
            f"{asset_j} rel %": sampled[asset_j],
            "rel diff %": sampled[asset_i] - sampled[asset_j],
            "daily diff %": relative[asset_i].diff().sub(relative[asset_j].diff()).fillna(0.0).iloc[positions],
        },
        index=sampled.index,
    )
    return table.T


def plot_relative_price_panel(
    axis,
    relative: pd.DataFrame,
    pair_row: pd.Series,
    candidate_dates: pd.Series,
) -> None:
    """Plot normalized two-asset price movement."""
    asset_i = relative.columns[0]
    asset_j = relative.columns[1]
    axis.plot(relative.index, relative[asset_i], label=f"{asset_i} rel %", linewidth=1.8)
    axis.plot(relative.index, relative[asset_j], label=f"{asset_j} rel %", linewidth=1.8)
    for candidate_date in candidate_dates:
        axis.axvline(candidate_date, color="black", alpha=0.22, linewidth=1.0)
    axis.axhline(0.0, color="gray", linewidth=0.8, alpha=0.55)
    title = (
        f"{pair_row['pair']} | selected {int(pair_row['selection_count'])}x | "
        f"max gap {float(pair_row['max_gap']):.3f}"
    )
    axis.set_title(title)
    axis.set_ylabel("Relative movement (%)")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)


def plot_transition_table_panel(
    axis,
    transition_table: pd.DataFrame,
    candidate_dates: pd.Series,
) -> None:
    """Plot a heatmap that reads like a movement transition table."""
    values = transition_table.to_numpy(dtype=float)
    limit = max(1.0, float(np.nanmax(np.abs(values))))
    image = axis.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    axis.set_yticks(range(len(transition_table.index)))
    axis.set_yticklabels(transition_table.index)
    axis.set_xticks(range(len(transition_table.columns)))
    axis.set_xticklabels([date.strftime("%Y-%m-%d") for date in transition_table.columns], rotation=90, fontsize=7)
    mark_candidate_columns(axis, transition_table.columns, candidate_dates)
    axis.set_title("Movement transition table")
    plt.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="%")


def mark_candidate_columns(axis, table_dates: pd.Index, candidate_dates: pd.Series) -> None:
    """Mark columns nearest to relationship-gap candidate dates."""
    date_positions = pd.Series(range(len(table_dates)), index=pd.to_datetime(table_dates))
    for candidate_date in pd.to_datetime(candidate_dates):
        nearest_position = int(np.argmin(np.abs((date_positions.index - candidate_date).days)))
        axis.axvline(nearest_position, color="black", alpha=0.35, linewidth=1.0)


def figure_size(transition_table: pd.DataFrame) -> tuple[float, float]:
    """Return a readable figure size for the number of sampled date columns."""
    width = min(22.0, max(11.0, len(transition_table.columns) * 0.28))
    return width, 8.0


def pair_image_filename(pair_row: pd.Series) -> str:
    """Return a stable filename for a pair image."""
    rank = int(pair_row["pair_set_rank"])
    safe_pair = str(pair_row["pair"]).replace("/", "_").replace(" ", "_")
    return f"{rank:03d}_{safe_pair}.png"
