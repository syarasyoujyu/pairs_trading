"""Model configurations mirroring the paper's forecasting experiments."""
from common.models import ARMAForecastConfig, EncoderDecoderForecastConfig, LSTMForecastConfig


def paper_arma_configs() -> tuple[ARMAForecastConfig, ...]:
    """Return the ARMA settings highlighted in the paper's Table 4."""
    return (
        ARMAForecastConfig(ar_order=5, ma_order=2, horizon=1),
        ARMAForecastConfig(ar_order=8, ma_order=3, horizon=1),
        ARMAForecastConfig(ar_order=12, ma_order=4, horizon=1),
    )


def best_paper_arma_config() -> ARMAForecastConfig:
    """Return the ARMA configuration emphasized by validation MSE in Table 4."""
    return paper_arma_configs()[1]


def paper_lstm_configs() -> tuple[LSTMForecastConfig, ...]:
    """Return representative LSTM settings from the paper's Table 4."""
    return (
        LSTMForecastConfig(input_length=12, hidden_layers=1, hidden_units=10, horizon=1),
        LSTMForecastConfig(input_length=24, hidden_layers=1, hidden_units=50, horizon=1),
        LSTMForecastConfig(input_length=24, hidden_layers=1, hidden_units=60, horizon=1),
    )


def best_paper_lstm_config() -> LSTMForecastConfig:
    """Return the LSTM configuration emphasized by validation MSE in Table 4."""
    return paper_lstm_configs()[1]


def paper_encoder_decoder_configs() -> tuple[EncoderDecoderForecastConfig, ...]:
    """Return representative encoder-decoder settings from the paper."""
    return (
        EncoderDecoderForecastConfig(input_length=12, encoder_units=30, decoder_units=30, horizon=1),
        EncoderDecoderForecastConfig(input_length=12, encoder_units=30, decoder_units=30, horizon=2),
        EncoderDecoderForecastConfig(input_length=24, encoder_units=15, decoder_units=15, horizon=1),
        EncoderDecoderForecastConfig(input_length=24, encoder_units=15, decoder_units=15, horizon=2),
        EncoderDecoderForecastConfig(input_length=24, encoder_units=30, decoder_units=30, horizon=1),
        EncoderDecoderForecastConfig(input_length=24, encoder_units=30, decoder_units=30, horizon=2),
    )


def best_paper_encoder_decoder_config() -> EncoderDecoderForecastConfig:
    """Return the two-step encoder-decoder configuration emphasized in Table 4."""
    return paper_encoder_decoder_configs()[3]
