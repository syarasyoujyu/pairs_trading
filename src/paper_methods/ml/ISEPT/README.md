# ISEPT: Image-Based Selection and Execution Framework for Pair Trading

`references/ml/Image-Based Selection and Execution Framework for Pair.pdf` と `dudskrla/ISEPT` を参考にした、Modal GPU 実行対応の実装です。

ISEPT は、OHLC を candlestick 画像に変換し、CAE で銘柄 latent vector を作り、2銘柄の latent vector を連結した pair representation から翌月 Sharpe ratio を MLP で予測します。選ばれたペアは Vidyamurthy 型の cointegration spread trading で6か月売買し、その実現 Sharpe を次回以降の MLP 学習ラベルに戻します。

VIDYAMURTHYとGATEVの売買ルールの詳細は [`TRADING_METHODS.md`](TRADING_METHODS.md) に分けています。

## 実行方法

デフォルトは `ISEPT + VIDYAMURTHY` で、論文に示された主要設定に寄せています。`--trade-rule` で執行ルールを選べます。

```bash
uv run src/paper_methods/ml/ISEPT/run.py --trade-rule vidyamurthy --max-assets 20 --months 8
```

GATEV執行で比較する場合:

```bash
uv run src/paper_methods/ml/ISEPT/run.py --trade-rule gatev --max-assets 20 --months 8
```

Modal GPU の疎通と処理全体だけを軽く確認する場合:

```bash
uv run src/paper_methods/ml/ISEPT/run.py --trade-rule vidyamurthy --max-assets 6 --months 4 --cae-epochs 1 --mlp-epochs 2 --batch-size 64 --top-k-pairs 3 --feedback-pairs-per-side 5 --output-name ISEPT_vidyamurthy_modal_smoke
```

GATEV執行の軽量確認:

```bash
uv run src/paper_methods/ml/ISEPT/run.py --trade-rule gatev --max-assets 6 --months 4 --cae-epochs 1 --mlp-epochs 2 --batch-size 64 --top-k-pairs 3 --feedback-pairs-per-side 5 --output-name ISEPT_gatev_modal_smoke
```

フルに近づける場合は、`data/raw/{symbol}/1d.csv` に S&P 500 構成銘柄を用意し、`--max-assets` と `--months` を広げて実行します。
`--min-history-years` は銘柄選定前に要求する最低履歴年数で、デフォルトは `8.0` です。短期上場銘柄が混ざって共通期間が短くなることを防ぎます。

## 論文設定

画像:

- 各ペア選定月について、直近 12 か月相当の `252` 営業日を使う。
- `21` 営業日の candlestick window を `1` 日ずつスライドする。
- 各windowを log scaling した `64 x 64 x 3` RGB画像にする。
- CAE用画像は `70%` train、`30%` validation にランダム分割する。

CAE:

- input: `64 x 64 x 3` candlestick image
- encoder: `64 -> 128 -> 256` channels、各段 `Conv2D / BatchNorm / PReLU / MaxPool`
- latent: `8 x 8 x 256 = 16,384` 次元
- decoder: `256 -> 128 -> 64 -> 3` channels
- loss: MSE
- optimizer: Adam
- learning rate: `1e-4`
- 最大 `20` epochs
- `5` epochs ごとに learning rate を半減
- validation loss が `3` 回連続で改善しない場合 early stopping

Pair MLP:

- 銘柄ごとに、CAE latent を月内画像で平均する。
- ペア表現は `[z_i, z_j]` の連結で `32,768` 次元。
- PCAで `512` 次元へ圧縮する。
- network: `LayerNorm -> Dense(1024) -> Dense(512) -> Dense(128) -> Dense(1)`
- hidden activation: ReLU
- dropout: `0.5`
- output: 翌月の予測 Sharpe ratio
- loss: MSE
- optimizer: Adam
- learning rate: `1e-3`
- batch size: `512`
- validation loss が `5` 回連続で改善しない場合 early stopping

選定とフィードバック:

- 各月、候補ペアを予測 Sharpe ratio で順位付けする。
- 上位 `100` ペアを売買対象にする。
- 過去月の実現 Sharpe ratio から top `20` と bottom `20` のペアをMLP教師データに戻す。

Trading:

- デフォルトでは ISEPT + VIDYAMURTHY の売買ルールを使う。
- formation: 月末以前 `252` 本で log spread を推定。
- trade window: 選定月の翌月から `6` か月。
- spread: `log(price_i) - gamma * log(price_j) - intercept`。
- threshold: `Δ = 0.75σ`。
- entry: spread が `mean - Δ` 以下なら long spread、`mean + Δ` 以上なら short spread。
- exit: long spread は上側band、short spread は下側bandに到達したら close。
- transaction cost: one-way `1bp`。

`--trade-rule gatev` を指定した場合は、GATEV型の `±2σ` entry / `±1σ` exit を使います。

## ワークフロー

1. `data/raw/{symbol}/1d.csv` から OHLCV を読む。
2. 平均 dollar volume で対象銘柄を絞る。
3. 各月末について、直近 `252` 本から `21` 営業日 candlestick 画像を1日刻みで作る。
4. 画像 tensor を Modal GPU に送り、CAE を MSE reconstruction loss で学習する。
5. CAE encoder 出力を月・銘柄ごとに平均し、stock latent vector にする。
6. 全候補ペアの実現 trading をローカルで計算し、実現 Sharpe ラベルを作る。
7. Modal 側で、過去月の top/bottom Sharpe ペアを MLP の training set にする。
8. pair embedding をPCAで圧縮し、MLPで翌月 Sharpe を予測する。
9. predicted Sharpe 上位ペアを選ぶ。
10. 選ばれたペアを設定された `trade_rule` でシミュレーションする。

## 出力

```text
data/results/{output_name}/{trade_rule}/cae_mlp/
├── candlestick_image_metadata.csv
├── pair_feedback_labels.csv
├── selected_pairs.csv
├── signals.csv
├── trades.csv
├── portfolio_daily_returns.csv
├── pair_daily_traces.csv
├── pair_daily_trace_manifest.csv
├── pair_daily_traces/
├── pair_diagnostic_plots/
├── trading_metrics.csv
├── run_config.csv
└── model/
    └── cae_history.csv
```

`selected_pairs.csv`:

- `month_end`
- `asset_i`, `asset_j`
- `predicted_sharpe`
- `rank`

`pair_feedback_labels.csv`:

- MLP の教師データ候補
- 全候補ペアの実現tradingによる `realized_sharpe`
- `trade_count`, `total_return_pct`

`signals.csv`:

- 選定ペアの spread, signal, entry/exit threshold

`trades.csv`:

- open/close episode 単位の trade record

`pair_daily_traces.csv`:

- selected event ごとの日次トレースを縦持ちで保存する
- 対象期間は、データロード後に残った共通履歴の先頭から、選定後 `trading_horizon_months` の売買window終了まで
- `model_split_period`: `train`, `validation`, `test`
- ISEPT は CAE 画像の train / validation split がランダムなので、この診断では選定月末までの可視化履歴の前半70%を `train`、後半30%を `validation`、選定後の売買期間を `test` として表示する
- `event_phase`: `lookback_train`, `lookback_validation`, `selection_date`, `trading_test`
- `raw_pair_daily_return`: signal を掛ける前の hedge-adjusted pair return
- `strategy_daily_return`, `strategy_cumulative_return`: 実際の signal と transaction cost を反映した日次/累積return
- `signal`: `1` は long spread、`-1` は short spread、`0` は flat
- `trade_action`, `trade_marker`: いつ long / short を建てたか、いつ閉じたか
- `spread`, `spread_z`, `spread_mean`, `spread_std`: 売買ルールで使う spread 指標
- VIDYAMURTHY では `lower_band`, `upper_band`, `delta`
- GATEV では `long_entry`, `short_entry`, `long_exit`, `short_exit`

`pair_daily_traces/{event_id}.csv`:

- `pair_daily_traces.csv` を selected event ごとに分割したCSV

`pair_diagnostic_plots/{event_id}.png`:

- 上段: 2銘柄の相対価格推移。緑線が trade open、赤線が trade close
- 2段目: spread と売買threshold
- 3段目: signal
- 下段: 日次returnと累積return
- 背景色は緑が train、黄土色が validation、青が test を表す
- trade marker の緑線は open、赤線は close を表し、背景色の train / validation / test とは別の意味を持つ

`pair_daily_trace_manifest.csv`:

- selected event ごとの trace CSV と plot PNG のパス
- `trade_count`, `total_strategy_return_pct`, `trace_start_date`, `trace_end_date`

`trading_metrics.csv`:

- `total_return_pct`
- `sharpe_ratio`
- `max_drawdown_pct`
- `volatility`
- `hit_ratio`
- `trade_count`
- `avg_holding_days`

## 注意

- 論文の完全再現には、S&P 500 構成銘柄の長期OHLCデータと長い評価期間が必要です。この実装は `data/raw` にある手元データを使って同じワークフローを実行します。
- `ISEPT + VIDYAMURTHY` と比較用の `ISEPT + GATEV` を実装しています。
- `--max-assets` を大きくし、`--window-step 1` のまま実行すると画像数が急増します。Modal実行時間と転送量に注意してください。
