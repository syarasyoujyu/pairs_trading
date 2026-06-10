"""Monthly calendar helpers for ISEPT."""
import pandas as pd


def month_end_dates(index: pd.Index, lookback_bars: int, forward_bars: int, months: int) -> list[pd.Timestamp]:
    """Return usable month-end selection dates from a trading index."""
    dates = pd.Index(pd.to_datetime(index)).sort_values()
    month_ends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).max().tolist()
    usable = []
    for date in month_ends:
        position = int(dates.get_loc(date))
        if position < lookback_bars:
            continue
        if position + forward_bars >= len(dates):
            continue
        usable.append(pd.Timestamp(date))
    return usable[-months:]


def next_month_window(index: pd.Index, month_end: pd.Timestamp) -> pd.Index:
    """Return trading dates after month_end through the next month-end."""
    return future_months_window(index, month_end, 1)


def future_months_window(index: pd.Index, month_end: pd.Timestamp, horizon_months: int) -> pd.Index:
    """Return trading dates after month_end through the requested calendar horizon."""
    dates = pd.Index(pd.to_datetime(index)).sort_values()
    start_position = int(dates.get_loc(month_end)) + 1
    if start_position >= len(dates):
        return pd.Index([])
    start_date = dates[start_position]
    month_period = start_date.to_period("M")
    end_period = month_period + max(1, horizon_months) - 1
    periods = pd.Series(dates, index=dates).dt.to_period("M")
    mask = (periods >= month_period) & (periods <= end_period)
    return dates[mask.to_numpy()]
