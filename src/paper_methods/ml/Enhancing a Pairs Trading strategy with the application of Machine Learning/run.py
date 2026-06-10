"""Run Sarmento and Horta (2020) on the SPY holdings universe.

Usage:
    uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py"
    uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py" --forecast-model lstm --neural-backend modal
"""
from argparse import ArgumentParser
from contextlib import nullcontext
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.models import PairCandidate
from data_loading.prices import filter_price_history, load_price_frame, load_spy_universe_symbols
from execution.outputs import save_comparison, save_daily_returns, save_pair_candidates, save_pair_summary
from execution.performance import (
    benchmark_daily_returns,
    pair_daily_returns,
    pair_trade_returns,
    portfolio_daily_returns,
    summarize_paper_trading_metrics,
    summarize_returns,
)
from forecasting_models.forecast_metrics import forecast_error_metrics, horizon_target, summarize_forecast_error_rows
from forecasting_models.forecasting import rolling_ar_forecast, rolling_arma_forecast
from forecasting_models.neural_prediction import predict_neural_forecaster
from forecasting_models.neural_training import fit_encoder_decoder_forecaster, fit_lstm_forecaster
from forecasting_models.paper_model_configs import (
    best_paper_arma_config,
    best_paper_encoder_decoder_config,
    best_paper_lstm_config,
)
from modal_execution.client import modal_app_context, train_and_predict_neural_forecaster
from pair_selection.features import normalized_returns
from pair_selection.selection import diagnose_pair_candidates
from pair_selection.spreads import build_spread
from trading.signals import generate_forecast_signals, generate_standard_signals, signals_to_trades
from trading.thresholds import forecast_threshold_candidates, standard_thresholds
from trading.validation import choose_forecast_thresholds
from visualization.cluster_movement import save_cluster_return_movement_plots
from visualization.pair_diagnostics import (
    prepare_pair_diagnostic_dirs,
    save_pair_diagnostic_outputs,
    save_pair_diagnostic_summary,
)
from visualization.pair_price_movement import save_pair_price_movement_plot
from visualization.trade_timing import save_trade_timing_plot


INTERVAL = "1d"
N_COMPONENTS = 5
MIN_SAMPLES = 5
CLUSTER_METHOD = "auto"
BARS_PER_DAY = 1
BARS_PER_YEAR = 252
FORMATION_BARS = 756
TRADING_BARS = 252
VALIDATION_RATIO = 0.2
MAX_MISSING_RATIO = 0.01
MAX_PAIRS_PER_CLUSTER = 300
MAX_SELECTED_PAIRS = 20
ARMA_CONFIG = best_paper_arma_config()
ROLLING_AR_ORDER = 5
ROLLING_AR_HORIZON = 1
FORECAST_MODEL = "arma"  # arma, rolling_ar, lstm, encoder_decoder
NEURAL_BACKEND = "modal"  # modal, local
MODAL_SAVE_MODELS = True
LSTM_CONFIG = best_paper_lstm_config()
ENCODER_DECODER_CONFIG = best_paper_encoder_decoder_config()

DATA_DIR = _ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
UNIVERSE_PATH = DATA_DIR / "universe" / "spy_holdings.csv"
OUTPUT_DIR = DATA_DIR / "results" / "EnhancingPairsTradingML" / FORECAST_MODEL
SIGNALS_DIR = DATA_DIR / "signals" / "EnhancingPairsTradingML" / FORECAST_MODEL
PLOTS_DIR = OUTPUT_DIR / "plots"


def main() -> None:
    args = parse_args()
    apply_runtime_args(args)
    print(f"[LOAD] universe={UNIVERSE_PATH}")
    symbols = load_spy_universe_symbols(UNIVERSE_PATH, include_spy=True)
    prices = load_price_frame(RAW_DIR, INTERVAL, symbols=symbols, drop_missing_rows=False)
    analysis_prices = _prepare_analysis_prices(prices)
    pair_prices = analysis_prices.drop(columns=["spy"], errors="ignore")
    benchmark_price = analysis_prices["spy"]

    print(
        f"[DATA] assets={pair_prices.shape[1]} bars={len(analysis_prices)} "
        f"{analysis_prices.index[0].date()} ~ {analysis_prices.index[-1].date()}"
    )

    formation_prices = pair_prices.iloc[:FORMATION_BARS]
    trading_prices = pair_prices.iloc[FORMATION_BARS:]
    pair_candidates, labels = diagnose_pair_candidates(
        formation_prices,
        n_components=N_COMPONENTS,
        min_samples=MIN_SAMPLES,
        cluster_method=CLUSTER_METHOD,
        bars_per_day=BARS_PER_DAY,
        bars_per_year=BARS_PER_YEAR,
        max_pairs_per_cluster=MAX_PAIRS_PER_CLUSTER,
    )
    selected_pairs = _selected_pairs(pair_candidates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUTPUT_DIR / "cluster_labels.csv")
    save_pair_candidates(pair_candidates, OUTPUT_DIR / "candidate_diagnostics.csv")
    save_pair_candidates(selected_pairs, OUTPUT_DIR / "selected_pairs.csv")
    cluster_plot_rows = save_cluster_return_movement_plots(
        _clustering_input_returns(formation_prices),
        labels,
        PLOTS_DIR / "clusters",
    )
    save_comparison(cluster_plot_rows, OUTPUT_DIR / "cluster_plot_manifest.csv")
    print(f"[SELECT] candidates={len(pair_candidates)} selected={len(selected_pairs)}")

    if not selected_pairs:
        save_comparison([], OUTPUT_DIR / "portfolio_comparison.csv")
        print("[DONE] No pair passed the paper's filters.")
        return

    standard_returns: dict[str, pd.Series] = {}
    forecast_returns: dict[str, pd.Series] = {}
    standard_trade_returns: dict[str, pd.Series] = {}
    forecast_trade_returns: dict[str, pd.Series] = {}
    summary_rows: list[dict] = []
    pair_paper_metric_rows: list[dict] = []
    forecast_metric_rows: list[dict] = []
    diagnostic_frames: list[pd.DataFrame] = []
    diagnostic_manifest_rows: list[dict] = []
    diagnostic_trace_dir, diagnostic_plot_dir = prepare_pair_diagnostic_dirs(OUTPUT_DIR)

    with _forecast_runtime_context():
        for pair in selected_pairs:
            pair_key = f"{pair.asset_y}_{pair.asset_x}"
            print(f"[TRADE] {pair_key}")
            full_spread = _build_full_spread(pair, pair_prices)
            formation_spread = full_spread.iloc[:FORMATION_BARS]
            trading_spread = full_spread.iloc[FORMATION_BARS:]

            standard_signals, standard_trades = _standard_trade(pair, formation_spread, trading_spread)
            forecast_signals, forecast_trades, pair_forecast_metrics = _forecast_trade(
                pair_key,
                pair,
                full_spread,
                formation_spread,
                trading_spread,
            )
            forecast_metric_rows.extend(pair_forecast_metrics)

            _save_pair_outputs(pair_key, standard_signals, standard_trades, forecast_signals, forecast_trades)
            _save_pair_visualizations(pair_key, pair, pair_prices, standard_signals, forecast_signals)
            diagnostic_frame, diagnostic_manifest_row = save_pair_diagnostic_outputs(
                pair_key,
                pair,
                pair_prices,
                full_spread,
                standard_signals,
                standard_trades,
                forecast_signals,
                forecast_trades,
                pair_prices.index[min(_validation_start_bar(), len(pair_prices.index) - 1)],
                pair_prices.index[min(FORMATION_BARS, len(pair_prices.index) - 1)],
                FORECAST_MODEL,
                diagnostic_trace_dir,
                diagnostic_plot_dir,
            )
            diagnostic_frames.append(diagnostic_frame)
            diagnostic_manifest_rows.append(diagnostic_manifest_row)
            standard_returns[pair_key] = pair_daily_returns(
                standard_signals,
                trading_prices[pair.asset_y],
                trading_prices[pair.asset_x],
            )
            forecast_returns[pair_key] = pair_daily_returns(
                forecast_signals,
                trading_prices[pair.asset_y],
                trading_prices[pair.asset_x],
            )
            standard_trade_returns[pair_key] = pair_trade_returns(
                standard_signals,
                trading_prices[pair.asset_y],
                trading_prices[pair.asset_x],
            )
            forecast_trade_returns[pair_key] = pair_trade_returns(
                forecast_signals,
                trading_prices[pair.asset_y],
                trading_prices[pair.asset_x],
            )
            summary_rows.extend(_pair_summary_rows(pair_key, pair, standard_returns[pair_key], forecast_returns[pair_key]))
            pair_paper_metric_rows.extend(
                _pair_paper_metric_rows(
                    pair_key,
                    pair,
                    standard_returns[pair_key],
                    forecast_returns[pair_key],
                    standard_trade_returns[pair_key],
                    forecast_trade_returns[pair_key],
                )
            )

    save_pair_diagnostic_summary(diagnostic_frames, diagnostic_manifest_rows, OUTPUT_DIR)
    standard_portfolio = portfolio_daily_returns(standard_returns)
    forecast_portfolio = portfolio_daily_returns(forecast_returns)
    benchmark = benchmark_daily_returns(benchmark_price.reindex(standard_portfolio.index.union(forecast_portfolio.index)))
    comparison_rows = [
        _portfolio_comparison_row("strategy", "standard_threshold", standard_portfolio),
        _portfolio_comparison_row("strategy", f"forecast_{FORECAST_MODEL}", forecast_portfolio),
        _portfolio_comparison_row("benchmark", "SPY buy-and-hold", benchmark),
    ]
    paper_metric_rows = [
        {"role": "strategy", **summarize_paper_trading_metrics(
            "standard_threshold",
            standard_portfolio,
            standard_returns,
            standard_trade_returns,
            BARS_PER_YEAR,
        )},
        {"role": "strategy", **summarize_paper_trading_metrics(
            f"forecast_{FORECAST_MODEL}",
            forecast_portfolio,
            forecast_returns,
            forecast_trade_returns,
            BARS_PER_YEAR,
        )},
        {"role": "benchmark", **summarize_paper_trading_metrics(
            "SPY buy-and-hold",
            benchmark,
            periods_per_year=BARS_PER_YEAR,
        )},
    ]

    save_pair_summary(summary_rows, OUTPUT_DIR / "pair_summary.csv")
    save_comparison(pair_paper_metric_rows, OUTPUT_DIR / "pair_trading_metrics.csv")
    save_comparison(comparison_rows, OUTPUT_DIR / "portfolio_comparison.csv")
    save_comparison(paper_metric_rows, OUTPUT_DIR / "paper_trading_metrics.csv")
    forecast_summary_rows = summarize_forecast_error_rows(forecast_metric_rows)
    save_comparison(forecast_metric_rows, OUTPUT_DIR / "forecast_error_metrics.csv")
    save_comparison(forecast_summary_rows, OUTPUT_DIR / "forecast_error_summary.csv")
    save_daily_returns(
        {
            "standard_threshold": standard_portfolio,
            f"forecast_{FORECAST_MODEL}": forecast_portfolio,
            "benchmark_spy_buy_hold": benchmark,
        },
        OUTPUT_DIR / "portfolio_daily_returns.csv",
    )
    print(f"[SAVE] {OUTPUT_DIR}")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    if forecast_summary_rows:
        print(pd.DataFrame(forecast_summary_rows).to_string(index=False))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--forecast-model", choices=["arma", "rolling_ar", "lstm", "encoder_decoder"], default=FORECAST_MODEL)
    parser.add_argument("--neural-backend", choices=["modal", "local"], default=NEURAL_BACKEND)
    parser.add_argument("--max-selected-pairs", type=int, default=MAX_SELECTED_PAIRS)
    parser.add_argument("--output-name", default="EnhancingPairsTradingML")
    parser.add_argument("--modal-save-models", action="store_true", default=MODAL_SAVE_MODELS)
    parser.add_argument("--no-modal-save-models", action="store_false", dest="modal_save_models")
    return parser.parse_args()


def apply_runtime_args(args) -> None:
    global FORECAST_MODEL, MAX_SELECTED_PAIRS, MODAL_SAVE_MODELS, NEURAL_BACKEND, OUTPUT_DIR, PLOTS_DIR, SIGNALS_DIR
    FORECAST_MODEL = args.forecast_model
    NEURAL_BACKEND = args.neural_backend
    MAX_SELECTED_PAIRS = args.max_selected_pairs
    MODAL_SAVE_MODELS = args.modal_save_models
    OUTPUT_DIR = result_output_dir(DATA_DIR / "results", args.output_name, FORECAST_MODEL, NEURAL_BACKEND)
    SIGNALS_DIR = result_output_dir(DATA_DIR / "signals", args.output_name, FORECAST_MODEL, NEURAL_BACKEND)
    PLOTS_DIR = OUTPUT_DIR / "plots"


def result_output_dir(base_dir: Path, output_name: str, forecast_model: str, neural_backend: str) -> Path:
    """Return a nested output directory for a forecast model run."""
    if forecast_model in {"lstm", "encoder_decoder"}:
        return base_dir / output_name / forecast_model / neural_backend
    return base_dir / output_name / forecast_model


def _prepare_analysis_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required_bars = FORMATION_BARS + TRADING_BARS
    recent = _benchmark_aligned_prices(prices).tail(required_bars)
    filtered = filter_price_history(
        recent,
        min_non_null=required_bars,
        max_missing_ratio=MAX_MISSING_RATIO,
    )
    if "spy" not in filtered.columns:
        raise ValueError("SPY benchmark data is required in data/raw/spy.")
    if len(filtered) < required_bars:
        raise ValueError(f"Need at least {required_bars} common bars, got {len(filtered)}.")
    return filtered.iloc[-required_bars:]


def _benchmark_aligned_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if "spy" not in prices.columns:
        raise ValueError("SPY benchmark data is required in data/raw/spy.")
    return prices.sort_index().loc[prices["spy"].notna()]


def _clustering_input_returns(prices: pd.DataFrame) -> pd.DataFrame:
    clean_prices = prices.astype(float).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return normalized_returns(clean_prices)


def _selected_pairs(pair_candidates: list[PairCandidate]) -> list[PairCandidate]:
    selected = [candidate for candidate in pair_candidates if candidate.diagnostics.is_selected]
    selected.sort(key=lambda candidate: candidate.diagnostics.adf_t_stat)
    return selected[:MAX_SELECTED_PAIRS]


def _build_full_spread(pair: PairCandidate, prices: pd.DataFrame) -> pd.Series:
    dependent = np.log(prices[pair.asset_y].astype(float))
    independent = np.log(prices[pair.asset_x].astype(float))
    return build_spread(dependent, independent, pair.hedge_ratio, pair.intercept)


def _standard_trade(
    pair: PairCandidate,
    formation_spread: pd.Series,
    trading_spread: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = standard_thresholds(formation_spread)
    signals = generate_standard_signals(trading_spread, thresholds)
    trades = signals_to_trades(signals, pair.asset_y, pair.asset_x, pair.hedge_ratio)
    return signals, trades


def _forecast_trade(
    pair_key: str,
    pair: PairCandidate,
    full_spread: pd.Series,
    formation_spread: pd.Series,
    trading_spread: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    validation_start = _validation_start_bar()
    threshold_source = formation_spread.iloc[:validation_start]
    validation_spread = formation_spread.iloc[validation_start:]
    validation_prediction, predicted, forecast_horizon = _forecast_predictions(
        full_spread=full_spread,
        formation_spread=formation_spread,
        validation_spread=validation_spread,
        trading_spread=trading_spread,
        validation_start=validation_start,
        pair_key=pair_key,
    )
    candidates = forecast_threshold_candidates(threshold_source, horizon=forecast_horizon)
    thresholds = choose_forecast_thresholds(validation_spread, validation_prediction, candidates)
    signals = generate_forecast_signals(trading_spread, predicted, thresholds)
    trades = signals_to_trades(signals, pair.asset_y, pair.asset_x, pair.hedge_ratio)
    metrics = _forecast_metric_rows(
        pair_key,
        formation_spread,
        full_spread,
        validation_prediction,
        predicted,
        forecast_horizon,
    )
    return signals, trades, metrics


def _forecast_predictions(
    full_spread: pd.Series,
    formation_spread: pd.Series,
    validation_spread: pd.Series,
    trading_spread: pd.Series,
    validation_start: int,
    pair_key: str,
) -> tuple[pd.Series, pd.Series, int]:
    if FORECAST_MODEL == "arma":
        validation_prediction = rolling_arma_forecast(
            formation_spread,
            ar_order=ARMA_CONFIG.ar_order,
            ma_order=ARMA_CONFIG.ma_order,
            horizon=ARMA_CONFIG.horizon,
            min_train=validation_start,
        ).reindex(validation_spread.index)
        predicted = rolling_arma_forecast(
            full_spread,
            ar_order=ARMA_CONFIG.ar_order,
            ma_order=ARMA_CONFIG.ma_order,
            horizon=ARMA_CONFIG.horizon,
            min_train=FORMATION_BARS,
        ).reindex(trading_spread.index)
        return validation_prediction, predicted, ARMA_CONFIG.horizon

    if FORECAST_MODEL == "rolling_ar":
        validation_prediction = rolling_ar_forecast(
            formation_spread,
            order=ROLLING_AR_ORDER,
            horizon=ROLLING_AR_HORIZON,
            min_train=validation_start,
            window=validation_start,
        ).reindex(validation_spread.index)
        predicted = rolling_ar_forecast(
            full_spread,
            order=ROLLING_AR_ORDER,
            horizon=ROLLING_AR_HORIZON,
            min_train=FORMATION_BARS,
            window=FORMATION_BARS,
        ).reindex(trading_spread.index)
        return validation_prediction, predicted, ROLLING_AR_HORIZON

    if FORECAST_MODEL == "lstm":
        if NEURAL_BACKEND == "modal":
            return _modal_neural_predictions(
                "lstm",
                pair_key,
                full_spread,
                formation_spread,
                validation_spread,
                trading_spread,
                validation_start,
                LSTM_CONFIG,
            )
        print(f"[TRAIN] per-pair LSTM model for {pair_key}")
        forecaster = fit_lstm_forecaster(formation_spread.iloc[:validation_start], validation_spread, LSTM_CONFIG)
        validation_prediction = predict_neural_forecaster(forecaster, formation_spread).reindex(validation_spread.index)
        predicted = predict_neural_forecaster(forecaster, full_spread).reindex(trading_spread.index)
        return validation_prediction, predicted, LSTM_CONFIG.horizon

    if FORECAST_MODEL == "encoder_decoder":
        if NEURAL_BACKEND == "modal":
            return _modal_neural_predictions(
                "encoder_decoder",
                pair_key,
                full_spread,
                formation_spread,
                validation_spread,
                trading_spread,
                validation_start,
                ENCODER_DECODER_CONFIG,
            )
        print(f"[TRAIN] per-pair encoder-decoder model for {pair_key}")
        forecaster = fit_encoder_decoder_forecaster(
            formation_spread.iloc[:validation_start],
            validation_spread,
            ENCODER_DECODER_CONFIG,
        )
        validation_prediction = predict_neural_forecaster(forecaster, formation_spread).reindex(validation_spread.index)
        predicted = predict_neural_forecaster(forecaster, full_spread).reindex(trading_spread.index)
        return validation_prediction, predicted, ENCODER_DECODER_CONFIG.horizon

    raise ValueError("FORECAST_MODEL must be 'arma', 'rolling_ar', 'lstm', or 'encoder_decoder'.")


def _modal_neural_predictions(
    model_kind: str,
    pair_key: str,
    full_spread: pd.Series,
    formation_spread: pd.Series,
    validation_spread: pd.Series,
    trading_spread: pd.Series,
    validation_start: int,
    config,
) -> tuple[pd.Series, pd.Series, int]:
    model_id = f"{pair_key}_{model_kind}_{formation_spread.index[0].date()}_{formation_spread.index[-1].date()}"
    print(f"[MODAL TRAIN] per-pair {model_kind} model_id={model_id}")
    result = train_and_predict_neural_forecaster(
        model_kind=model_kind,
        train_spread=formation_spread.iloc[:validation_start],
        validation_spread=validation_spread,
        prediction_spreads={
            "validation": formation_spread,
            "trading": full_spread,
        },
        config=config,
        model_id=model_id,
        save_model=MODAL_SAVE_MODELS,
    )
    validation_prediction = result["predictions"]["validation"].reindex(validation_spread.index)
    predicted = result["predictions"]["trading"].reindex(trading_spread.index)
    return validation_prediction, predicted, config.horizon


def _forecast_runtime_context():
    if _uses_modal_neural_backend():
        return modal_app_context()
    return nullcontext()


def _uses_modal_neural_backend() -> bool:
    return FORECAST_MODEL in {"lstm", "encoder_decoder"} and NEURAL_BACKEND == "modal"


def _save_pair_outputs(
    pair_key: str,
    standard_signals: pd.DataFrame,
    standard_trades: pd.DataFrame,
    forecast_signals: pd.DataFrame,
    forecast_trades: pd.DataFrame,
) -> None:
    standard_signals.to_csv(SIGNALS_DIR / f"{pair_key}_standard_signals.csv", index=False)
    standard_trades.to_csv(SIGNALS_DIR / f"{pair_key}_standard_trades.csv", index=False)
    forecast_signals.to_csv(SIGNALS_DIR / f"{pair_key}_{FORECAST_MODEL}_signals.csv", index=False)
    forecast_trades.to_csv(SIGNALS_DIR / f"{pair_key}_{FORECAST_MODEL}_trades.csv", index=False)


def _save_pair_visualizations(
    pair_key: str,
    pair: PairCandidate,
    prices: pd.DataFrame,
    standard_signals: pd.DataFrame,
    forecast_signals: pd.DataFrame,
) -> None:
    pair_plot_dir = PLOTS_DIR / "pairs" / pair_key
    validation_start = prices.index[min(_validation_start_bar(), len(prices.index) - 1)]
    trading_start = prices.index[min(FORMATION_BARS, len(prices.index) - 1)]
    save_pair_price_movement_plot(
        pair_key,
        pair.asset_y,
        pair.asset_x,
        prices[pair.asset_y],
        prices[pair.asset_x],
        standard_signals,
        forecast_signals,
        validation_start,
        trading_start,
        pair_plot_dir / f"{pair_key}_price_movement.png",
    )
    save_trade_timing_plot(
        pair_key,
        "standard_threshold",
        standard_signals,
        pair_plot_dir / f"{pair_key}_standard_trade_timing.png",
    )
    save_trade_timing_plot(
        pair_key,
        f"forecast_{FORECAST_MODEL}",
        forecast_signals,
        pair_plot_dir / f"{pair_key}_{FORECAST_MODEL}_trade_timing.png",
    )


def _validation_start_bar() -> int:
    return int(FORMATION_BARS * (1.0 - VALIDATION_RATIO))


def _pair_summary_rows(
    pair_key: str,
    pair: PairCandidate,
    standard_returns: pd.Series,
    forecast_returns: pd.Series,
) -> list[dict]:
    standard = summarize_returns("standard_threshold", standard_returns, BARS_PER_YEAR)
    forecast = summarize_returns(f"forecast_{FORECAST_MODEL}", forecast_returns, BARS_PER_YEAR)
    rows = []
    for row in (standard, forecast):
        rows.append(
            {
                "pair": pair_key,
                "asset_y": pair.asset_y,
                "asset_x": pair.asset_x,
                "hedge_ratio": pair.hedge_ratio,
                "adf_t_stat": pair.diagnostics.adf_t_stat,
                "hurst": pair.diagnostics.hurst,
                "half_life_bars": pair.diagnostics.half_life_bars,
                **row,
            }
        )
    return rows


def _pair_paper_metric_rows(
    pair_key: str,
    pair: PairCandidate,
    standard_returns: pd.Series,
    forecast_returns: pd.Series,
    standard_trade_returns: pd.Series,
    forecast_trade_returns: pd.Series,
) -> list[dict]:
    rows = []
    for name, returns, trade_returns in (
        ("standard_threshold", standard_returns, standard_trade_returns),
        (f"forecast_{FORECAST_MODEL}", forecast_returns, forecast_trade_returns),
    ):
        rows.append(
            {
                "pair": pair_key,
                "asset_y": pair.asset_y,
                "asset_x": pair.asset_x,
                "hedge_ratio": pair.hedge_ratio,
                **summarize_paper_trading_metrics(
                    name,
                    returns,
                    {pair_key: returns},
                    {pair_key: trade_returns},
                    BARS_PER_YEAR,
                ),
            }
        )
    return rows


def _forecast_metric_rows(
    pair_key: str,
    formation_spread: pd.Series,
    full_spread: pd.Series,
    validation_prediction: pd.Series,
    trading_prediction: pd.Series,
    horizon: int,
) -> list[dict]:
    return [
        _forecast_metric_row(pair_key, "validation", formation_spread, validation_prediction, horizon),
        _forecast_metric_row(pair_key, "test", full_spread, trading_prediction, horizon),
    ]


def _forecast_metric_row(
    pair_key: str,
    period: str,
    spread: pd.Series,
    predicted: pd.Series,
    horizon: int,
) -> dict:
    metrics = forecast_error_metrics(horizon_target(spread, horizon), predicted)
    return {
        "pair": pair_key,
        "model": FORECAST_MODEL,
        "period": period,
        "horizon": horizon,
        **_forecast_config_row(),
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mse_e03": metrics["mse"] * 1_000,
        "rmse_e02": metrics["rmse"] * 100,
        "mae_e02": metrics["mae"] * 100,
    }


def _forecast_config_row() -> dict:
    if FORECAST_MODEL == "arma":
        return {"ar_order": ARMA_CONFIG.ar_order, "ma_order": ARMA_CONFIG.ma_order}
    if FORECAST_MODEL == "rolling_ar":
        return {"ar_order": ROLLING_AR_ORDER, "ma_order": 0}
    if FORECAST_MODEL == "lstm":
        return {
            "input_length": LSTM_CONFIG.input_length,
            "hidden_layers": LSTM_CONFIG.hidden_layers,
            "hidden_units": LSTM_CONFIG.hidden_units,
        }
    if FORECAST_MODEL == "encoder_decoder":
        return {
            "input_length": ENCODER_DECODER_CONFIG.input_length,
            "encoder_units": ENCODER_DECODER_CONFIG.encoder_units,
            "decoder_units": ENCODER_DECODER_CONFIG.decoder_units,
        }
    return {}


def _portfolio_comparison_row(role: str, name: str, daily_returns: pd.Series) -> dict:
    row = summarize_returns(name, daily_returns, BARS_PER_YEAR)
    return {"role": role, "name": row.pop("model"), **row}


if __name__ == "__main__":
    main()
