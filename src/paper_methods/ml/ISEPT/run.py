"""Run ISEPT with Modal-backed CAE/MLP training and pair trading.

Usage:
    uv run src/paper_methods/ml/ISEPT/run.py --max-assets 12 --months 6 --output-name ISEPT
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.config import ImageConfig, ModalModelConfig, RuntimeConfig, TradingConfig
from data_loading.prices import close_frame, load_ohlcv_panel
from execution.outputs import save_frame, save_metrics, save_series
from images.calendar import month_end_dates
from images.dataset import available_symbols_by_month, build_monthly_image_payload
from labels.pair_labels import realized_pair_labels_by_month
from modal_execution.client import modal_app_context, modal_gpu_status, train_select_pairs_on_modal
from trading.metrics import trading_metrics
from trading.pair_diagnostics import save_pair_diagnostics
from trading.simulation import simulate_selected_pairs


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
        months=args.months,
        random_seed=args.seed,
    )
    image_config = ImageConfig(
        image_size=args.image_size,
        candle_window=args.candle_window,
        window_step=args.window_step,
        lookback_bars=args.lookback_bars,
    )
    model_config = ModalModelConfig(
        cae_latent_dim=args.cae_latent_dim,
        cae_base_filters=args.cae_base_filters,
        cae_epochs=args.cae_epochs,
        cae_learning_rate=args.cae_learning_rate,
        cae_lr_decay_epochs=args.cae_lr_decay_epochs,
        cae_lr_decay_factor=args.cae_lr_decay_factor,
        cae_patience=args.cae_patience,
        pca_components=args.pca_components,
        mlp_epochs=args.mlp_epochs,
        mlp_learning_rate=args.mlp_learning_rate,
        mlp_patience=args.mlp_patience,
        batch_size=args.batch_size,
        feedback_pairs_per_side=args.feedback_pairs_per_side,
        warmup_months=args.warmup_months,
        top_k_pairs=args.top_k_pairs,
        allow_asset_reuse=args.allow_asset_reuse,
    )
    trading_config = TradingConfig(
        trade_rule=args.trade_rule,
        trading_horizon_months=args.trading_horizon_months,
        entry_sigma=args.entry_sigma,
        exit_sigma=args.exit_sigma,
        vidyamurthy_threshold_sigma=args.vidyamurthy_threshold_sigma,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    output_dir = result_output_dir(RESULTS_DIR, runtime, trading_config)

    print(
        f"[LOAD] raw_dir={RAW_DIR} interval={runtime.interval} "
        f"max_assets={runtime.max_assets} min_history_years={runtime.min_history_years}"
    )
    panel = load_ohlcv_panel(
        RAW_DIR,
        runtime.interval,
        runtime.max_assets,
        exclude_symbols={"spy"},
        min_history_years=runtime.min_history_years,
    )
    close = close_frame(panel)
    print(f"[DATA] assets={len(panel)} bars={len(close)} {close.index[0].date()} ~ {close.index[-1].date()}")

    forward_bars = trading_config.trading_horizon_months * 23
    selection_months = month_end_dates(close.index, image_config.lookback_bars, forward_bars, runtime.months)
    if len(selection_months) <= model_config.warmup_months:
        raise ValueError("months must be greater than warmup_months.")
    print(f"[MONTHS] count={len(selection_months)} {selection_months[0].date()} ~ {selection_months[-1].date()}")

    images, image_metadata = build_monthly_image_payload(panel, selection_months, image_config)
    symbols_by_month = available_symbols_by_month(image_metadata)
    labels = realized_pair_labels_by_month(panel, selection_months, symbols_by_month, image_config, trading_config)
    save_frame(pd.DataFrame(image_metadata), output_dir / "candlestick_image_metadata.csv")
    save_frame(labels, output_dir / "pair_feedback_labels.csv")
    print(f"[IMAGES] count={len(images)} shape={images.shape[1:]}")
    print(f"[LABELS] rows={len(labels)}")

    with modal_app_context():
        status = modal_gpu_status()
        print(f"[MODAL] tensorflow={status['tensorflow_version']} gpu_count={status['gpu_count']}")
        result = train_select_pairs_on_modal(images, image_metadata, labels, model_config, runtime.random_seed)

    selected_pairs = result["selected_pairs"]
    save_frame(selected_pairs, output_dir / "selected_pairs.csv")
    save_frame(pd.DataFrame(result["cae_history"]), output_dir / "model" / "cae_history.csv")
    print(f"[SELECT] selected_events={len(selected_pairs)} embeddings={result['embedding_count']}")

    portfolio_returns, signals, trades = simulate_selected_pairs(selected_pairs, panel, image_config, trading_config)
    save_series(portfolio_returns, output_dir / "portfolio_daily_returns.csv")
    save_frame(signals, output_dir / "signals.csv")
    save_frame(trades, output_dir / "trades.csv")
    pair_diagnostic_manifest = save_pair_diagnostics(
        selected_pairs,
        panel,
        signals,
        trades,
        image_config,
        trading_config,
        output_dir,
    )
    print(f"[PAIR_DIAGNOSTICS] saved={len(pair_diagnostic_manifest)}")
    metrics = trading_metrics(portfolio_returns, trades)
    save_metrics(metrics, output_dir / "trading_metrics.csv")
    save_run_config(runtime, image_config, model_config, trading_config, output_dir / "run_config.csv")

    print(f"[SAVE] {output_dir}")
    print(pd.DataFrame([metrics]).to_string(index=False))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--output-name", default="ISEPT")
    parser.add_argument("--max-assets", type=int, default=20)
    parser.add_argument("--min-history-years", type=float, default=8.0)
    parser.add_argument("--months", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--candle-window", type=int, default=21)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--lookback-bars", type=int, default=252)
    parser.add_argument("--cae-latent-dim", type=int, default=16_384)
    parser.add_argument("--cae-base-filters", type=int, default=64)
    parser.add_argument("--cae-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", "--cae-learning-rate", dest="cae_learning_rate", type=float, default=0.0001)
    parser.add_argument("--cae-lr-decay-epochs", type=int, default=5)
    parser.add_argument("--cae-lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--cae-patience", type=int, default=3)
    parser.add_argument("--pca-components", type=int, default=512)
    parser.add_argument("--mlp-epochs", type=int, default=20)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.001)
    parser.add_argument("--mlp-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feedback-pairs-per-side", type=int, default=20)
    parser.add_argument("--warmup-months", type=int, default=2)
    parser.add_argument("--top-k-pairs", type=int, default=100)
    parser.set_defaults(allow_asset_reuse=True)
    parser.add_argument("--allow-asset-reuse", action="store_true")
    parser.add_argument("--no-asset-reuse", action="store_false", dest="allow_asset_reuse")
    parser.add_argument("--trade-rule", choices=["gatev", "vidyamurthy"], default="vidyamurthy")
    parser.add_argument("--trading-horizon-months", type=int, default=6)
    parser.add_argument("--entry-sigma", type=float, default=2.0)
    parser.add_argument("--exit-sigma", type=float, default=1.0)
    parser.add_argument("--vidyamurthy-threshold-sigma", type=float, default=0.75)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    return parser.parse_args()


def result_output_dir(results_dir: Path, runtime: RuntimeConfig, trading_config: TradingConfig) -> Path:
    """Return the nested result directory for an ISEPT trade-rule run."""
    return results_dir / runtime.output_name / trading_config.trade_rule / "cae_mlp"


def save_run_config(
    runtime: RuntimeConfig,
    image: ImageConfig,
    model: ModalModelConfig,
    trading: TradingConfig,
    output_path: Path,
) -> None:
    """Persist run configuration."""
    rows = []
    for section_name, config in (
        ("runtime", runtime),
        ("image", image),
        ("model", model),
        ("trading", trading),
    ):
        for key, value in vars(config).items():
            rows.append({"section": section_name, "key": key, "value": value})
    save_frame(pd.DataFrame(rows), output_path)


if __name__ == "__main__":
    main()
