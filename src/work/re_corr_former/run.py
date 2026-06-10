"""Run the ReCorrFormer research-plan pipeline.

Usage:
    uv run src/work/re_corr_former/run.py --backend local --max-assets 40
    uv run src/work/re_corr_former/run.py --backend modal --max-assets 12 --epochs 1
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.config import CandidateConfig, ModelConfig, RuntimeConfig, ScoringConfig, SharpeModelConfig, TradeConfig, WindowConfig
from data_loading.market_data import close_volume_from_panel, load_ohlcv_panel
from evaluation.metrics import pair_selection_metrics, trading_metrics
from execution.outputs import save_frame, save_metrics, save_series
from features.asset_features import build_asset_features
from pairs.candidate_reports import candidate_pairs_to_frame, summarize_candidate_pairs
from pairs.candidate_visuals import candidate_pair_set_frame, save_candidate_pair_movement_images
from pairs.candidates import generate_candidate_pairs
from selection.scoring import add_final_score, select_pairs_by_date
from sharpe_prediction.datasets import build_sharpe_dataset
from sharpe_prediction.metrics import sharpe_selection_metrics
from sharpe_prediction.rolling import rolling_sharpe_predictions, select_sharpe_pairs_by_month
from trading.pair_diagnostics import save_pair_diagnostics
from trading.rule_backtest import backtest_rule_pairs
from training.datasets import add_high_corr_label_columns, build_supervised_dataset, normalize_split_scalars, split_supervised_dataset
from training.train_model import predict_re_corr_former, train_re_corr_former


DATA_DIR = _ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results"


def main() -> None:
    args = parse_args()
    runtime = RuntimeConfig(
        interval=args.interval,
        output_name=args.output_name,
        max_assets=args.max_assets,
        min_history_years=args.min_history_years,
        random_seed=args.seed,
        backend=args.backend,
        selection_model=args.selection_model,
    )
    windows = WindowConfig(
        lookback=args.lookback,
        long_corr_window=args.long_corr_window,
        long_corr_min_lag=args.long_corr_min_lag,
        short_corr_window=args.short_corr_window,
        short_corr_min_lag=args.short_corr_min_lag,
        future_min_horizon=args.future_min_horizon,
        future_max_horizon=args.future_max_horizon,
        future_corr_window=args.future_corr_window,
        sample_stride=args.sample_stride,
    )
    candidates_config = CandidateConfig(
        min_long_corr=args.min_long_corr,
        long_corr_top_fraction=args.long_corr_top_fraction,
        min_gap=args.min_gap,
        max_candidates_per_date=args.max_candidates_per_date,
    )
    model_config = ModelConfig(
        encoder_units=args.encoder_units,
        dense_units=args.dense_units,
        dense_layers=args.dense_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    sharpe_model_config = SharpeModelConfig(
        model_type=args.sharpe_model,
        lookback=args.sharpe_lookback,
        encoder_units=args.sharpe_encoder_units,
        dense_units=args.sharpe_dense_units,
        dense_layers=args.sharpe_dense_layers,
        transformer_heads=args.sharpe_transformer_heads,
        transformer_ff_dim=args.sharpe_transformer_ff_dim,
        transformer_layers=args.sharpe_transformer_layers,
        dropout=args.sharpe_dropout,
        learning_rate=args.sharpe_learning_rate,
        batch_size=args.sharpe_batch_size,
        epochs=args.sharpe_epochs,
        patience=args.sharpe_patience,
        warmup_months=args.sharpe_warmup_months,
        feedback_pairs_per_side=args.sharpe_feedback_pairs_per_side,
    )
    scoring_config = ScoringConfig(
        top_k=args.top_k,
        allow_asset_reuse=args.allow_asset_reuse,
        high_corr_top_fraction=args.high_corr_top_fraction,
    )
    trade_config = TradeConfig(
        trade_rule=args.trade_rule,
        formation_window=args.formation_window,
        trading_horizon_months=args.trading_horizon_months,
        entry_sigma=args.entry_sigma,
        exit_sigma=args.exit_sigma,
        vidyamurthy_threshold_sigma=args.vidyamurthy_threshold_sigma,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    output_dir = result_output_dir(RESULTS_DIR, runtime, trade_config, sharpe_model_config)

    np.random.seed(runtime.random_seed)
    print(
        f"[LOAD] raw_dir={RAW_DIR} interval={runtime.interval} "
        f"max_assets={runtime.max_assets} min_history_years={runtime.min_history_years}"
    )
    panel = load_ohlcv_panel(
        RAW_DIR,
        runtime.interval,
        runtime.max_assets,
        min_history_years=runtime.min_history_years,
        exclude_symbols={"spy"},
    )
    close, volume = close_volume_from_panel(panel)
    print(f"[DATA] assets={close.shape[1]} bars={len(close)} {close.index[0].date()} ~ {close.index[-1].date()}")

    features = build_asset_features(close, volume)
    returns = close.astype(float).pct_change().dropna()
    print("[CANDIDATES] relationship-gap scan")
    candidates = generate_candidate_pairs(close, volume, windows, candidates_config)
    if not candidates:
        raise ValueError("No relationship-gap candidates were generated. Increase long_corr_top_fraction or relax min_gap.")
    print(f"[CANDIDATES] count={len(candidates)}")
    candidate_frame = candidate_pairs_to_frame(candidates, candidates_config)
    candidate_summary = summarize_candidate_pairs(candidate_frame)
    pair_set = candidate_pair_set_frame(candidate_frame)
    save_frame(candidate_frame, output_dir / "relationship_gap_candidates.csv")
    save_frame(candidate_summary, output_dir / "relationship_gap_candidate_summary.csv")
    save_frame(pair_set, output_dir / "relationship_gap_pair_set.csv")
    if args.candidate_images:
        image_manifest = save_candidate_pair_movement_images(
            pair_set,
            candidate_frame,
            close,
            output_dir / "relationship_gap_pair_movements",
            windows,
            args.candidate_image_limit,
            args.pair_movement_table_step,
        )
        save_frame(image_manifest, output_dir / "relationship_gap_pair_movement_images.csv")
        print(f"[CANDIDATE_IMAGES] saved={len(image_manifest)} pair_set={len(pair_set)}")
    print_candidate_preview(candidate_frame, args.candidate_preview_rows)
    if args.stop_after_candidates:
        save_run_config(runtime, windows, candidates_config, model_config, sharpe_model_config, scoring_config, trade_config, output_dir / "run_config.csv")
        print(f"[SAVE] {output_dir}")
        return

    dataset = build_supervised_dataset(candidates, features, close, returns, windows)
    dataset = add_high_corr_label_columns(dataset, scoring_config.high_corr_top_fraction)
    save_frame(dataset.metadata, output_dir / "candidate_labels.csv")
    predictions, selected, selection_row = select_pairs(dataset, panel, close, windows, model_config, sharpe_model_config, scoring_config, trade_config, runtime, output_dir)
    save_frame(selected, output_dir / "selected_pairs.csv")
    portfolio_returns, trades, trade_signals = backtest_rule_pairs(
        selected,
        close,
        trade_config,
    )
    save_series(portfolio_returns, output_dir / "portfolio_daily_returns.csv")
    save_frame(trades, output_dir / "trades.csv")
    save_frame(trade_signals, output_dir / "signals.csv")
    if args.pair_diagnostics:
        pair_manifest = save_pair_diagnostics(
            selected,
            close,
            predictions,
            dataset.metadata,
            trade_signals,
            trades,
            windows,
            trade_config,
            model_config.validation_fraction,
            output_dir,
            args.pair_diagnostic_limit,
        )
        print(f"[PAIR_DIAGNOSTICS] saved={len(pair_manifest)}")

    trading_row = trading_metrics(portfolio_returns, trades)
    save_metrics(selection_row, output_dir / "selection_metrics.csv")
    save_metrics(trading_row, output_dir / "trading_metrics.csv")
    save_run_config(runtime, windows, candidates_config, model_config, sharpe_model_config, scoring_config, trade_config, output_dir / "run_config.csv")

    print(f"[SAVE] {output_dir}")
    print(pd.DataFrame([selection_row]).to_string(index=False))
    print(pd.DataFrame([trading_row]).to_string(index=False))


def parse_args():
    model_defaults = ModelConfig()
    sharpe_defaults = SharpeModelConfig()
    parser = ArgumentParser()
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--selection-model", choices=["corr", "sharpe"], default="corr")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--output-name", default="ReCorrFormer")
    parser.add_argument("--max-assets", type=int, default=40)
    parser.add_argument("--min-history-years", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--long-corr-window", type=int, default=120)
    parser.add_argument("--long-corr-min-lag", type=int, default=15)
    parser.add_argument("--short-corr-window", type=int, default=15)
    parser.add_argument("--short-corr-min-lag", type=int, default=1)
    parser.add_argument("--future-min-horizon", type=int, default=2)
    parser.add_argument("--future-max-horizon", type=int, default=30)
    parser.add_argument("--future-corr-window", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--min-long-corr", type=float, default=-1.0)
    parser.add_argument("--long-corr-top-fraction", type=float, default=0.15)
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument("--max-candidates-per-date", type=int, default=80)
    parser.add_argument("--candidate-preview-rows", type=int, default=10)
    parser.add_argument("--stop-after-candidates", action="store_true")
    parser.add_argument("--candidate-images", action="store_true", default=True)
    parser.add_argument("--no-candidate-images", action="store_false", dest="candidate_images")
    parser.add_argument("--candidate-image-limit", type=int, default=40)
    parser.add_argument("--pair-movement-table-step", type=int, default=5)
    parser.add_argument("--encoder-units", type=int, default=model_defaults.encoder_units)
    parser.add_argument("--dense-units", type=int, default=model_defaults.dense_units)
    parser.add_argument("--dense-layers", type=int, default=model_defaults.dense_layers)
    parser.add_argument("--dropout", type=float, default=model_defaults.dropout)
    parser.add_argument("--learning-rate", type=float, default=model_defaults.learning_rate)
    parser.add_argument("--epochs", type=int, default=model_defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=model_defaults.batch_size)
    parser.add_argument("--sharpe-model", choices=["lstm", "transformer"], default=sharpe_defaults.model_type)
    parser.add_argument("--sharpe-lookback", type=int, default=sharpe_defaults.lookback)
    parser.add_argument("--sharpe-encoder-units", type=int, default=sharpe_defaults.encoder_units)
    parser.add_argument("--sharpe-dense-units", type=int, default=sharpe_defaults.dense_units)
    parser.add_argument("--sharpe-dense-layers", type=int, default=sharpe_defaults.dense_layers)
    parser.add_argument("--sharpe-transformer-heads", type=int, default=sharpe_defaults.transformer_heads)
    parser.add_argument("--sharpe-transformer-ff-dim", type=int, default=sharpe_defaults.transformer_ff_dim)
    parser.add_argument("--sharpe-transformer-layers", type=int, default=sharpe_defaults.transformer_layers)
    parser.add_argument("--sharpe-dropout", type=float, default=sharpe_defaults.dropout)
    parser.add_argument("--sharpe-learning-rate", type=float, default=sharpe_defaults.learning_rate)
    parser.add_argument("--sharpe-batch-size", type=int, default=sharpe_defaults.batch_size)
    parser.add_argument("--sharpe-epochs", type=int, default=sharpe_defaults.epochs)
    parser.add_argument("--sharpe-patience", type=int, default=sharpe_defaults.patience)
    parser.add_argument("--sharpe-warmup-months", type=int, default=sharpe_defaults.warmup_months)
    parser.add_argument("--sharpe-feedback-pairs-per-side", type=int, default=sharpe_defaults.feedback_pairs_per_side)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--allow-asset-reuse", action="store_true")
    parser.add_argument("--high-corr-top-fraction", type=float, default=0.15)
    parser.add_argument("--trade-rule", choices=["vidyamurthy", "gatev"], default="vidyamurthy")
    parser.add_argument("--formation-window", type=int, default=252)
    parser.add_argument("--trading-horizon-months", type=int, default=6)
    parser.add_argument("--entry-sigma", type=float, default=2.0)
    parser.add_argument("--exit-sigma", type=float, default=1.0)
    parser.add_argument("--vidyamurthy-threshold-sigma", type=float, default=0.75)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--pair-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-pair-diagnostics", action="store_false", dest="pair_diagnostics")
    parser.add_argument("--pair-diagnostic-limit", type=int, default=0)
    return parser.parse_args()


def select_pairs(
    dataset,
    panel: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    windows: WindowConfig,
    model_config: ModelConfig,
    sharpe_model_config: SharpeModelConfig,
    scoring_config: ScoringConfig,
    trade_config: TradeConfig,
    runtime: RuntimeConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select pairs with the requested prediction target."""
    if runtime.selection_model == "corr":
        return select_pairs_by_correlation_model(dataset, windows, model_config, scoring_config, runtime, output_dir)
    if runtime.selection_model == "sharpe":
        return select_pairs_by_sharpe_model(dataset, panel, close, sharpe_model_config, scoring_config, trade_config, runtime, output_dir)
    raise ValueError("selection_model must be 'corr' or 'sharpe'.")


def select_pairs_by_correlation_model(
    dataset,
    windows: WindowConfig,
    model_config: ModelConfig,
    scoring_config: ScoringConfig,
    runtime: RuntimeConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train the original correlation-recovery model and select pairs."""
    splits = split_supervised_dataset(dataset, model_config)
    splits, _ = normalize_split_scalars(splits)
    print(split_summary(splits))
    predictions = train_and_predict(splits, windows.lookback, model_config, runtime, output_dir)
    predictions = add_final_score(predictions, scoring_config)
    save_frame(predictions, output_dir / "predictions.csv")
    test_predictions = predictions[predictions["split"] == "test"].reset_index(drop=True)
    selected = select_pairs_by_date(test_predictions, scoring_config)
    selection_row = pair_selection_metrics(test_predictions, selected, scoring_config.top_k)
    return predictions, selected, selection_row


def select_pairs_by_sharpe_model(
    dataset,
    panel: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    sharpe_model_config: SharpeModelConfig,
    scoring_config: ScoringConfig,
    trade_config: TradeConfig,
    runtime: RuntimeConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train the OHLCV+correlation Sharpe model and select pairs."""
    print(f"[SHARPE_LABELS] trade_rule={trade_config.trade_rule}")
    sharpe_dataset = build_sharpe_dataset(dataset.metadata, panel, close, trade_config, sharpe_model_config)
    save_frame(sharpe_dataset.metadata, output_dir / "sharpe_feedback_labels.csv")
    print(f"[SHARPE_LABELS] rows={len(sharpe_dataset.metadata)}")
    if sharpe_dataset.metadata.empty:
        raise ValueError("No Sharpe labels were generated. Reduce sharpe_lookback or relax candidate settings.")
    print(f"[SHARPE_MODEL] model={sharpe_model_config.model_type} feedback_pairs_per_side={sharpe_model_config.feedback_pairs_per_side}")
    predictions = rolling_sharpe_predictions(sharpe_dataset, sharpe_model_config, scoring_config, runtime.random_seed, output_dir)
    if predictions.empty:
        raise ValueError("No Sharpe predictions were generated. Reduce sharpe_warmup_months or increase candidate history.")
    save_frame(predictions, output_dir / "predictions.csv")
    save_frame(predictions, output_dir / "sharpe_predictions.csv")
    selected = select_sharpe_pairs_by_month(predictions, scoring_config)
    selection_row = sharpe_selection_metrics(predictions, selected)
    save_metrics(selection_row, output_dir / "sharpe_selection_metrics.csv")
    return predictions, selected, selection_row


def result_output_dir(
    results_dir: Path,
    runtime: RuntimeConfig,
    trade_config: TradeConfig,
    sharpe_model_config: SharpeModelConfig,
) -> Path:
    """Return the nested result directory for a method/trade/model run."""
    return results_dir / runtime.output_name / trade_config.trade_rule / selection_output_slug(runtime, sharpe_model_config)


def selection_output_slug(runtime: RuntimeConfig, sharpe_model_config: SharpeModelConfig) -> str:
    """Return a compact model variant slug for result paths."""
    if runtime.selection_model == "sharpe":
        return f"sharpe_{sharpe_model_config.model_type}"
    return "corr_re_corr_former"


def train_and_predict(
    splits: dict,
    lookback: int,
    model_config: ModelConfig,
    runtime: RuntimeConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Train and predict with the requested backend."""
    if runtime.backend == "local":
        model = train_re_corr_former(splits, lookback, model_config, runtime.random_seed, output_dir / "model")
        return predict_all_splits(model, splits)
    if runtime.backend == "modal":
        from modal_execution.client import modal_app_context, modal_gpu_status, train_predict_re_corr_former_on_modal

        with modal_app_context():
            status = modal_gpu_status()
            print(f"[MODAL] tensorflow={status['tensorflow_version']} gpu_count={status['gpu_count']}")
            result = train_predict_re_corr_former_on_modal(splits, lookback, model_config, runtime.random_seed)
        save_training_history(result["history"], output_dir / "model" / "training_history.csv")
        return result["predictions"]
    raise ValueError("backend must be 'local' or 'modal'.")


def predict_all_splits(model, splits: dict) -> pd.DataFrame:
    """Predict train, validation, and test splits."""
    frames = []
    for split_name, split in splits.items():
        predictions = predict_re_corr_former(model, split)
        predictions["split"] = split_name
        frames.append(predictions)
    return pd.concat(frames, axis=0, ignore_index=True)


def split_summary(splits: dict) -> str:
    """Return a compact split summary string."""
    parts = [f"{name}={len(split.metadata)}" for name, split in splits.items()]
    return "[SPLIT] " + " ".join(parts)


def print_candidate_preview(candidate_frame: pd.DataFrame, preview_rows: int) -> None:
    """Print the top relationship-gap candidates for quick inspection."""
    if preview_rows <= 0 or candidate_frame.empty:
        return
    columns = ["date", "rank_by_gap_on_date", "pair", "rho_long", "rho_now", "gap", "spread_z"]
    preview = candidate_frame.loc[:, columns].head(preview_rows)
    print("[CANDIDATES] relationship-gap preview")
    print(preview.to_string(index=False))


def save_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Persist Modal training history as CSV."""
    save_frame(pd.DataFrame(history), output_path)


def save_run_config(
    runtime: RuntimeConfig,
    windows: WindowConfig,
    candidates: CandidateConfig,
    model: ModelConfig,
    sharpe_model: SharpeModelConfig,
    scoring: ScoringConfig,
    trade: TradeConfig,
    output_path: Path,
) -> None:
    """Persist run configuration as key-value rows."""
    rows = []
    for section_name, config in (
        ("runtime", runtime),
        ("windows", windows),
        ("candidate", candidates),
        ("model", model),
        ("sharpe_model", sharpe_model),
        ("scoring", scoring),
        ("trade", trade),
    ):
        for key, value in vars(config).items():
            rows.append({"section": section_name, "key": key, "value": value})
    save_frame(pd.DataFrame(rows), output_path)


if __name__ == "__main__":
    main()
