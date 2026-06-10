"""Validation experiments for forecast model comparison."""
import pandas as pd

from common.models import ARMAForecastConfig, EncoderDecoderForecastConfig, LSTMForecastConfig
from forecasting_models.forecast_metrics import forecast_error_metrics, horizon_target
from forecasting_models.forecasting import naive_forecast, rolling_ar_forecast, rolling_arma_forecast
from forecasting_models.neural_prediction import predict_neural_forecaster
from forecasting_models.neural_training import fit_encoder_decoder_forecaster, fit_lstm_forecaster


def compare_validation_forecasts(
    spread: pd.Series,
    validation_start: int,
    formation_end: int,
    arma_config: ARMAForecastConfig | None = None,
    ar_order: int = 5,
    ar_horizon: int = 1,
    lstm_config: LSTMForecastConfig | None = None,
    encoder_decoder_config: EncoderDecoderForecastConfig | None = None,
) -> pd.DataFrame:
    """Compare forecast errors on the validation slice."""
    train_spread = spread.iloc[:validation_start]
    formation_spread = spread.iloc[:formation_end]
    validation_index = spread.iloc[validation_start:formation_end].index
    rows: list[dict] = []

    rows.append(
        _score_prediction(
            "naive",
            ar_horizon,
            formation_spread,
            naive_forecast(formation_spread, horizon=ar_horizon).reindex(validation_index),
        )
    )
    rows.append(
        _score_prediction(
            "rolling_ar",
            ar_horizon,
            formation_spread,
            rolling_ar_forecast(
                formation_spread,
                order=ar_order,
                horizon=ar_horizon,
                min_train=validation_start,
                window=validation_start,
            ).reindex(validation_index),
        )
    )

    if arma_config is not None:
        rows.append(
            _score_prediction(
                "arma",
                arma_config.horizon,
                formation_spread,
                rolling_arma_forecast(
                    formation_spread,
                    ar_order=arma_config.ar_order,
                    ma_order=arma_config.ma_order,
                    horizon=arma_config.horizon,
                    min_train=validation_start,
                ).reindex(validation_index),
            )
        )

    if lstm_config is not None:
        forecaster = fit_lstm_forecaster(train_spread, spread.iloc[validation_start:formation_end], lstm_config)
        rows.append(
            _score_prediction(
                "lstm",
                lstm_config.horizon,
                formation_spread,
                predict_neural_forecaster(forecaster, formation_spread).reindex(validation_index),
            )
        )

    if encoder_decoder_config is not None:
        forecaster = fit_encoder_decoder_forecaster(
            train_spread,
            spread.iloc[validation_start:formation_end],
            encoder_decoder_config,
        )
        rows.append(
            _score_prediction(
                "encoder_decoder",
                encoder_decoder_config.horizon,
                formation_spread,
                predict_neural_forecaster(forecaster, formation_spread).reindex(validation_index),
            )
        )

    return pd.DataFrame(rows)


def _score_prediction(
    name: str,
    horizon: int,
    spread: pd.Series,
    predicted: pd.Series,
) -> dict:
    metrics = forecast_error_metrics(horizon_target(spread, horizon), predicted)
    return {"model": name, "horizon": horizon, **metrics}
