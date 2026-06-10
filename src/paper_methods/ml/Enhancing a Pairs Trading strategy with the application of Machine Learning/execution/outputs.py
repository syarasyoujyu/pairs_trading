"""Output helpers for the full paper-method run."""
from dataclasses import asdict
from pathlib import Path

import pandas as pd


def save_pair_candidates(pair_candidates: list, output_path: Path) -> None:
    rows = []
    for pair in pair_candidates:
        row = asdict(pair.diagnostics)
        row["asset_y"] = pair.asset_y
        row["asset_x"] = pair.asset_x
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_pair_summary(rows: list[dict], output_path: Path) -> None:
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_comparison(rows: list[dict], output_path: Path) -> None:
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_daily_returns(returns: dict[str, pd.Series], output_path: Path) -> None:
    frame = pd.concat(returns.values(), axis=1).fillna(0.0)
    frame.columns = list(returns.keys())
    frame.to_csv(output_path, index_label="datetime")
