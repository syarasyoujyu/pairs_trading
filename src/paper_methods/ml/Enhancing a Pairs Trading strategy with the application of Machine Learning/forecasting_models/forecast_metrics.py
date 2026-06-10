"""Forecast evaluation metrics."""
import numpy as np
import pandas as pd


def forecast_error_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Compute MSE, RMSE, and MAE for aligned spread forecasts."""
    aligned = pd.concat([actual.astype(float).rename("actual"), predicted.astype(float).rename("predicted")], axis=1)
    aligned = aligned.dropna()
    if aligned.empty:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    errors = aligned["actual"] - aligned["predicted"]
    mse = float(np.mean(errors.to_numpy() ** 2))
    mae = float(np.mean(np.abs(errors.to_numpy())))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae}


def horizon_target(spread: pd.Series, horizon: int) -> pd.Series:
    """Return actual S(t+horizon) indexed by current timestamp t."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    return spread.astype(float).shift(-horizon).rename("actual_spread")


def summarize_forecast_error_rows(rows: list[dict]) -> list[dict]:
    """Average pair-level forecast errors into paper-style summary rows."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []

    summary_rows: list[dict] = []
    for (model, period, horizon), group in frame.groupby(["model", "period", "horizon"], sort=False):
        summary_rows.append(
            {
                "model": model,
                "period": period,
                "horizon": horizon,
                "pair_count": int(group["pair"].nunique()),
                **_first_forecast_config_values(group),
                **_mean_forecast_metrics(group),
            }
        )
    return summary_rows


def _first_forecast_config_values(group: pd.DataFrame) -> dict:
    config_columns = (
        "ar_order",
        "ma_order",
        "input_length",
        "hidden_layers",
        "hidden_units",
        "encoder_units",
        "decoder_units",
    )
    values = {}
    for column in config_columns:
        if column not in group.columns:
            continue
        non_null = group[column].dropna()
        values[column] = "" if non_null.empty else non_null.iloc[0]
    return values


def _mean_forecast_metrics(group: pd.DataFrame) -> dict:
    metric_columns = ("mse", "rmse", "mae", "mse_e03", "rmse_e02", "mae_e02")
    return {column: float(group[column].mean()) for column in metric_columns if column in group.columns}
