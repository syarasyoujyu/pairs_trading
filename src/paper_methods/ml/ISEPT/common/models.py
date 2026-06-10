"""Data containers for ISEPT."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MonthlyAssetImages:
    month_end: pd.Timestamp
    symbol: str
    images: object


@dataclass(frozen=True)
class PairSharpeLabel:
    label_month: pd.Timestamp
    asset_i: str
    asset_j: str
    realized_sharpe: float
    trade_count: int
    total_return_pct: float


@dataclass(frozen=True)
class SelectedPair:
    month_end: pd.Timestamp
    asset_i: str
    asset_j: str
    predicted_sharpe: float
    rank: int
