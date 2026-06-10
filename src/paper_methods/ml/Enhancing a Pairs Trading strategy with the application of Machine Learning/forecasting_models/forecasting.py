"""Forecasting helpers for the forecasting-based trading model."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def naive_forecast(spread: pd.Series, horizon: int = 1) -> pd.Series:
    """Return the naive forecast S*(t+h) = S(t)."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    return pd.Series(spread.astype(float).to_numpy(), index=spread.index, name="predicted_spread")


def rolling_ar_forecast(
    spread: pd.Series,
    order: int = 5,
    horizon: int = 1,
    min_train: int | None = None,
    window: int | None = None,
) -> pd.Series:
    """Fit a rolling AR(order) model and recursively forecast horizon bars."""
    if order < 1:
        raise ValueError("order must be at least 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    values = spread.astype(float).to_numpy()
    predictions = np.full(len(values), np.nan)
    min_train_used = min_train or (order + 2)

    for current in range(min_train_used, len(values)):
        start = 0 if window is None else max(0, current + 1 - window)
        history = values[start:current + 1]
        if len(history) <= order + 1:
            continue
        coefficients = _fit_ar_coefficients(history, order)
        predictions[current] = _recursive_ar_prediction(history, coefficients, order, horizon)

    return pd.Series(predictions, index=spread.index, name="predicted_spread")


def rolling_arma_forecast(
    spread: pd.Series,
    ar_order: int = 8,
    ma_order: int = 3,
    horizon: int = 1,
    min_train: int | None = None,
) -> pd.Series:
    """Fit ARMA(p, q) once and update its state for one forecast per bar."""
    if ar_order < 1:
        raise ValueError("ar_order must be at least 1.")
    if ma_order < 0:
        raise ValueError("ma_order must be at least 0.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    values = spread.astype(float).to_numpy()
    predictions = np.full(len(values), np.nan)
    min_train_used = min_train or max(ar_order + ma_order + 10, 30)
    if len(values) <= min_train_used:
        return pd.Series(predictions, index=spread.index, name="predicted_spread")

    fitted = _fit_arma_results(values[:min_train_used], ar_order, ma_order)
    for current in range(min_train_used, len(values)):
        fitted = fitted.append([values[current]], refit=False)
        predictions[current] = _arma_horizon_prediction(fitted, horizon)

    return pd.Series(predictions, index=spread.index, name="predicted_spread")


def statsmodels_arma_forecast(
    train_spread: pd.Series,
    forecast_index: pd.Index,
    ar_order: int = 8,
    ma_order: int = 3,
) -> pd.Series:
    """Fit ARMA(p, q) via statsmodels and forecast the requested index."""
    fitted = _fit_arma_results(train_spread.astype(float).to_numpy(), ar_order, ma_order)
    forecast = fitted.forecast(steps=len(forecast_index))
    return pd.Series(forecast, index=forecast_index, name="predicted_spread")


def _fit_arma_results(values: np.ndarray, ar_order: int, ma_order: int):
    model = ARIMA(
        values,
        order=(ar_order, 0, ma_order),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(method_kwargs={"warn_convergence": False})


def _arma_horizon_prediction(fitted, horizon: int) -> float:
    forecast = fitted.forecast(steps=horizon)
    return float(forecast[horizon - 1])


def _fit_ar_coefficients(history: np.ndarray, order: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    target: list[float] = []
    for idx in range(order, len(history)):
        rows.append(np.r_[1.0, history[idx - order:idx][::-1]])
        target.append(float(history[idx]))
    x = np.vstack(rows)
    y = np.asarray(target, dtype=float)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _recursive_ar_prediction(
    history: np.ndarray,
    coefficients: np.ndarray,
    order: int,
    horizon: int,
) -> float:
    values = list(history.astype(float))
    for _ in range(horizon):
        lags = np.asarray(values[-order:][::-1], dtype=float)
        values.append(float(coefficients[0] + coefficients[1:] @ lags))
    return values[-1]
