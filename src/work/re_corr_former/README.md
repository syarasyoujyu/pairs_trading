# ReCorrFormer: Relation Recovery Network

`references/ml/recorrelation_dnn_pairs_research_plan.pdf` の研究計画書に基づく実装です。

目的は、現在も高相関なペアを探すのではなく、長期的には連動していたが直近では相関が低下しているペアを候補にし、その後 `2〜30` 時点以内に再相関化する可能性を DNN で予測することです。

## 実行方法

```bash
uv run src/work/re_corr_former/run.py
```

軽い疎通確認:

```bash
uv run src/work/re_corr_former/run.py --max-assets 12 --epochs 1 --sample-stride 30 --output-name ReCorrFormerSmoke
```

Modal GPU での軽い疎通確認:

```bash
uv run src/work/re_corr_former/run.py --backend modal --max-assets 12 --epochs 1 --sample-stride 30 --max-candidates-per-date 20 --top-k 5 --output-name ReCorrFormerModalSmoke
```

GATEV執行で比較する場合:

```bash
uv run src/work/re_corr_former/run.py --trade-rule gatev --max-assets 12 --epochs 1 --sample-stride 30 --max-candidates-per-date 20 --top-k 5
```

Sharpe ratio 予測モデルでペア選定する場合:

```bash
uv run src/work/re_corr_former/run.py --selection-model sharpe --sharpe-model lstm --trade-rule vidyamurthy --max-assets 20 --sharpe-epochs 3 --sample-stride 30 --max-candidates-per-date 40 --top-k 10
```

Transformer版の Sharpe ratio 予測モデル:

```bash
uv run src/work/re_corr_former/run.py --selection-model sharpe --sharpe-model transformer --trade-rule gatev --max-assets 20 --sharpe-epochs 3 --sample-stride 30 --max-candidates-per-date 40 --top-k 10
```

関係性ギャップによる候補ペアだけ確認:

```bash
uv run src/work/re_corr_former/run.py --max-assets 40 --stop-after-candidates --output-name ReCorrFormer_candidates
```

候補ペア set と、ペアごとの値動き遷移画像だけ軽く確認:

```bash
uv run src/work/re_corr_former/run.py --max-assets 12 --sample-stride 30 --max-candidates-per-date 20 --candidate-image-limit 10 --stop-after-candidates --output-name ReCorrFormer_pair_images
```

主な引数:

- `--backend`: `local` または `modal`
- `--max-assets`: 平均ドル出来高で使う上位銘柄数
- `--min-history-years`: 銘柄選定前に要求する最低履歴年数。デフォルト `8.0`。短期上場銘柄が混ざって共通期間が短くなることを防ぐ
- `--long-corr-window`: 長期相関に使う最大 lag。デフォルト `120`
- `--long-corr-min-lag`: 長期相関に使う最小 lag。デフォルト `15`
- `--short-corr-window`: 短期相関に使う最大 lag。デフォルト `15`
- `--short-corr-min-lag`: 短期相関に使う最小 lag。デフォルト `1`
- `--future-min-horizon`, `--future-max-horizon`: 将来相関回復を見る範囲。デフォルト `2〜30`
- `--future-corr-window`: 将来相関を測る小窓
- `--min-long-corr`: Top15% 条件に加えて使う長期相関の追加下限。デフォルト `-1.0`
- `--long-corr-top-fraction`: 候補にする長期相関上位割合。デフォルト `0.15`
- `--min-gap`: `rho_long - rho_now` の下限
- `--stop-after-candidates`: 候補ペア CSV を保存した時点で止める
- `--candidate-preview-rows`: 標準出力に表示する候補ペア行数
- `--candidate-images`, `--no-candidate-images`: 候補ペア set の値動き画像を保存するか
- `--candidate-image-limit`: 画像化する unique pair 数。`0` 以下なら全 unique pair
- `--pair-movement-table-step`: 値動き遷移表で何営業日ごとに列を作るか
- `--top-k`: 各テスト日時で選ぶペア数
- `--selection-model`: `corr` または `sharpe`。デフォルト `corr`
- `--epochs`: `corr` モデルのDNN学習エポック数
- `--encoder-units`: 共有 LSTM の隠れユニット数。デフォルト `16`
- `--dense-units`: pair representation 後の Dense ユニット数。デフォルト `32`
- `--dense-layers`: pair representation 後の Dense 層数。デフォルト `1`
- `--sharpe-model`: Sharpe予測モデル。`lstm` または `transformer`
- `--sharpe-lookback`: Sharpe予測モデルに入れるOHLCV本数。デフォルト `60`
- `--sharpe-epochs`: Sharpe予測モデルのrolling学習エポック数。デフォルト `5`
- `--sharpe-warmup-months`: ISEPT式rolling feedback前のウォームアップ月数。デフォルト `2`
- `--sharpe-feedback-pairs-per-side`: 各過去月から教師に戻す上位/下位Sharpeペア数。デフォルト `20`
- `--trade-rule`: `vidyamurthy` または `gatev`。デフォルト `vidyamurthy`
- `--formation-window`: 売買ルールのspread推定に使う過去本数。デフォルト `252`
- `--trading-horizon-months`: 選定後に売買する月数。デフォルト `6`
- `--entry-sigma`: GATEV の entry threshold。デフォルト `2.0`
- `--exit-sigma`: GATEV の exit threshold。デフォルト `1.0`
- `--vidyamurthy-threshold-sigma`: Vidyamurthy band の `Delta`。デフォルト `0.75`
- `--transaction-cost-bps`: one-way transaction cost。デフォルト `1`
- `--high-corr-top-fraction`: `y_corr` が高いとみなす上位割合。デフォルト `0.15`
- `--pair-diagnostics`, `--no-pair-diagnostics`: 選定ペアごとの日次トレースCSVと診断画像を保存するか。デフォルトは保存する
- `--pair-diagnostic-limit`: 日次診断を保存する selected event 数。`0` 以下なら全 selected event

デフォルトのモデルは、Modal で短時間に試せるように研究計画書の ReCorrFormer を薄くしたプロトタイプです。共有 LSTM は `16` units、Dense は `32` units の `1` 層、学習は `2` epochs にしています。精度検証を重視する場合は、`--encoder-units`, `--dense-units`, `--dense-layers`, `--epochs`, `--max-assets` を上げてください。

## ワークフロー

1. `data/raw/{symbol}/1d.csv` から close と volume を読む。
2. 平均ドル出来高で対象銘柄を絞る。
3. 各銘柄について、リターン、ボラティリティ、対数ドル出来高、移動平均乖離、モメンタムを作る。
4. 各判定時点で長期相関 `rho_long` と短期相関 `rho_now` を計算する。
   `rho_now` は判定日 `t` の `1〜15` 日前、`rho_long` は `15〜120` 日前のリターン相関です。
5. 流動性フィルタ後の全ペアから、`rho_long` が日付内 Top15% かつ `gap = rho_long - rho_now` が正の候補ペアを作る。
6. 候補ペアごとに、将来 `2〜30` 時点以内の最大相関上昇幅 `y_corr` を教師信号にする。
   `y_corr` が高い候補の定義は、日付内で `y_corr > 0` かつ Top15% 以内です。
7. `--selection-model corr` では ReCorrFormer を `y_corr` だけに対して学習する。
8. `--selection-model corr` では、テスト期間の候補に対して将来最大相関上昇幅 `pred_corr` だけを予測する。
9. `--selection-model sharpe` では、候補ペアごとに選択された `trade_rule` で将来期間を売買し、実現 `realized_sharpe` をラベルにする。
10. Sharpe予測モデルは ISEPT と同じく、過去月の realized Sharpe 上位/下位ペアを feedback 教師データに戻し、rolling に次月候補の `predicted_sharpe` を予測する。
11. `corr` では `final_score = pred_corr`、`sharpe` では `final_score = predicted_sharpe` としてペアをスコア化する。
12. `corr` は各日時、`sharpe` は各月で Top-K を選び、同一銘柄の重複を避ける。
13. ここから先は `src/paper_methods/ml/ISEPT` と同じく、`--trade-rule` で選んだ VIDYAMURTHY または GATEV の売買ルールにする。
14. 選定ペアごとに、選定日前 `252` 本で `log(P_i) - intercept - beta * log(P_j)` の spread を推定する。
15. 選定日の翌営業日から `6` か月を trading window にする。
16. VIDYAMURTHY では `Delta = 0.75 * spread_std` とし、下側bandで long spread、上側bandで short spread を建てる。
17. GATEV では `mean ± 2σ` で entry、`mean ± 1σ` で exit する。

## モデル入力と出力

銘柄入力:

- `asset_i_sequence`: shape `(samples, lookback, feature_count)`
- `asset_j_sequence`: shape `(samples, lookback, feature_count)`
- feature は `return`, `volatility`, `log_dollar_volume`, `ma_gap`, `momentum`

ペア入力:

- `rho_long`
- `rho_now`
- `gap`
- `spread_z`
- `spread_volatility`
- `beta`

DNN内部:

- 共有 LSTM で各銘柄を `z_i`, `z_j` に変換する。
- `[z_i, z_j, |z_i-z_j|, z_i*z_j, pair_scalars]` を pair representation にする。
- Dense network に通し、将来最大相関上昇幅だけを予測する。
- Modal スモーク向けのデフォルトでは LSTM `16` units、Dense `32` units x `1` 層の薄い構成を使う。

ペア推定モデルの出力:

- `corr`: 将来最大相関上昇幅

Sharpe予測モデルの入力:

- `asset_i_ohlcv`: shape `(samples, sharpe_lookback, 5)`
- `asset_j_ohlcv`: shape `(samples, sharpe_lookback, 5)`
- OHLC は lookback 先頭日の close を基準に相対化する。
- volume は lookback window 内の `log1p(volume)` を z-score 化する。
- `pair_scalars`: `rho_long`, `rho_now`, `gap`

Sharpe予測モデルの学習:

- `--sharpe-model lstm` では2銘柄に共有 LSTM encoder を使う。
- `--sharpe-model transformer` では2銘柄に共有 Transformer encoder を使う。
- 出力は `predicted_sharpe` 1つだけ。
- 各候補の教師ラベル `realized_sharpe` は、同じ `trade_rule` で将来 trading window を売買して計算する。
- ISEPT と同じく、過去月ごとに realized Sharpe 上位/下位 `--sharpe-feedback-pairs-per-side` 件を rolling training set にする。

ペアトレード執行:

- ペア選定後は、追加のLSTMやspread予測モデルを学習しない。
- ISEPT と同じく、formation window のspread平均・標準偏差だけでbandを作る。
- `signal=1` は spread long、`signal=-1` は spread short、`signal=0` は flat。
- VIDYAMURTHY では `lower_band`, `upper_band`, `delta`、GATEV では `long_entry`, `short_entry`, `long_exit`, `short_exit` が `signals.csv` に保存される。
- どちらのルールでも `gamma`, `beta`, `intercept`, `spread_mean`, `spread_std`, `trade_rule` を保存する。

## 出力ファイル

```text
data/results/{output_name}/{trade_rule}/{model_variant}/
├── candidate_labels.csv
├── relationship_gap_candidates.csv
├── relationship_gap_candidate_summary.csv
├── relationship_gap_pair_set.csv
├── relationship_gap_pair_movement_images.csv
├── relationship_gap_pair_movements/
├── predictions.csv
├── sharpe_feedback_labels.csv
├── sharpe_predictions.csv
├── sharpe_selection_metrics.csv
├── selected_pairs.csv
├── signals.csv
├── pair_daily_traces.csv
├── pair_daily_trace_manifest.csv
├── pair_daily_traces/
├── pair_diagnostic_plots/
├── portfolio_daily_returns.csv
├── trades.csv
├── selection_metrics.csv
├── trading_metrics.csv
├── run_config.csv
└── model/
    ├── re_corr_former.keras
    ├── training_history.csv
    └── sharpe_{lstm_or_transformer}_history.csv
```

`--backend modal` では学習・推論を Modal 側で完結させ、ローカルには `model/training_history.csv` と予測・選定・バックテスト結果を保存します。現状の Modal 経路では `.keras` モデル本体はローカル保存しません。

`model_variant` は `--selection-model corr` なら `corr_re_corr_former`、`--selection-model sharpe --sharpe-model transformer` なら `sharpe_transformer` です。例えば GATEV + Sharpe Transformer は `data/results/ReCorrFormer/gatev/sharpe_transformer/` に保存されます。

`relationship_gap_candidates.csv`:

- DNN に入る前の候補ペア一覧
- 流動性フィルタ後の日付内 `rho_long` Top15% かつ `gap = rho_long - rho_now > min_gap` を満たしたペア
- 日付ごとの `rank_by_gap_on_date`、`rho_long`, `rho_now`, `gap`, `spread_z`, `beta`
- `long_corr_rank`, `long_corr_rank_fraction`, `long_corr_pair_count`
- `long_corr_start_date`, `long_corr_end_date`, `short_corr_start_date`, `short_corr_end_date`
- `passed_liquidity_filter=True` は、候補生成前の流動性フィルタを通過したことを表す

`relationship_gap_candidate_summary.csv`:

- 日付ごとの候補数
- 平均 `rho_long`, 平均 `rho_now`, 平均 `gap`, 最大 `gap`
- その日の `gap` 最大ペア

`relationship_gap_pair_set.csv`:

- 候補に出たペアを重複なしの set として集約した一覧
- `selection_count`: 何回候補に出たか
- `first_selected_date`, `last_selected_date`
- `max_gap`, `mean_gap`, `max_rho_long`, `min_rho_now`

`relationship_gap_pair_movements/`:

- unique pair ごとの PNG
- 上段は、最初の表示日を `0%` とした2銘柄の相対価格推移
- 黒い縦線は、そのペアが関係性ギャップ候補に選ばれた日
- 下段は、日付列 x 値動き行の遷移表ヒートマップ
- 行は `{asset_i} rel %`, `{asset_j} rel %`, `rel diff %`, `daily diff %`

`relationship_gap_pair_movement_images.csv`:

- 画像化したペアと PNG パスの manifest

`candidate_labels.csv`:

- 候補ペア、関係性ギャップ、spread特徴、教師ラベル
- `y_corr` が研究計画書の相関回復ラベル
- `is_high_y_corr` は `y_corr > 0` かつ日付内 Top15% の候補

`predictions.csv`:

- `--selection-model corr` では train / validation / test すべての予測
- `pred_corr`: ReCorrFormer が予測した将来最大相関上昇幅
- `--selection-model sharpe` では rolling prediction 対象月の候補予測
- `predicted_sharpe`: OHLCV+相関モデルが予測した実現Sharpe
- `final_score`: ペア選定に使うスコア。`corr` では `pred_corr`、`sharpe` では `predicted_sharpe`

`sharpe_feedback_labels.csv`:

- Sharpe予測モデル用の全候補ラベル
- `realized_sharpe`: 選択した `trade_rule` で将来windowを売買した実現Sharpe
- `realized_total_return_pct`, `realized_trade_count`
- `label_month`: ISEPT式rolling feedbackで使う月bucket

`sharpe_predictions.csv`:

- rolling学習で予測対象になった候補
- `predicted_sharpe`, `realized_sharpe`, `final_score`

`sharpe_selection_metrics.csv`:

- `spearman_predicted_realized_sharpe`
- `selected_mean_realized_sharpe`
- `selected_mean_predicted_sharpe`
- `selected_mean_trade_count`

`selected_pairs.csv`:

- テスト期間で選ばれたペア
- `final_score` が最終選定スコア

`signals.csv`:

- 選定ペアごとの売買 signal
- 共通列は `spread`, `signal`, `trade_rule`
- VIDYAMURTHY列は `lower_band`, `upper_band`, `delta`
- GATEV列は `long_entry`, `short_entry`, `long_exit`, `short_exit`
- spread 定義に使う `gamma`, `beta`, `intercept`
- `spread_mean`, `spread_std`

`pair_daily_traces.csv`:

- selected event ごとの日次トレースを縦持ちで保存する
- 対象期間は、データロード後に残った共通履歴の先頭から、選定後 `trading_horizon_months` の売買window終了まで
- `model_split_period`: 候補生成時系列から見た `train`, `validation`, `test` の期間。`corr` では `predictions.csv` の split、Sharpe rolling 系では候補日付から同じ比率で復元した split を使う
- `event_phase`: `formation`, `selection_date`, `trading`
- `raw_pair_daily_return`: `ret_i - beta * ret_j` を hedge 比率で正規化した、ポジションを掛ける前の日次ペアリターン
- `strategy_daily_return`: 実際の `signal` を1日ラグで掛け、transaction cost を差し引いた日次戦略リターン
- `strategy_cumulative_return`: selected event 単位の日次戦略リターンを複利累積したもの
- `signal`: `1` は long spread、`-1` は short spread、`0` は flat
- `trade_action`, `trade_marker`: いつ long / short を建てたか、いつ閉じたか
- `spread`, `spread_z`, `spread_mean`, `spread_std`: 売買ルールで直接使う spread 指標
- `rho_long`, `rho_now`, `gap`: 各日で再計算した長期相関、短期相関、関係性ギャップ
- `selected_rho_long`, `selected_rho_now`, `selected_gap`, `selected_final_score`: ペア選定日に使われた値

`pair_daily_traces/{event_id}.csv`:

- `pair_daily_traces.csv` を selected event ごとに分割したCSV
- ペア単位で pandas や表計算ソフトから確認しやすい形

`pair_diagnostic_plots/{event_id}.png`:

- 上段: 2銘柄の相対価格推移。黒点線が選定日、緑線が trade open、赤線が trade close
- 2段目: `rho_long`, `rho_now`, `gap` の日次推移
- 3段目: spread、売買threshold、long / short action
- 下段: `strategy_daily_return` と `strategy_cumulative_return`
- 背景色は `model_split_period` に対応し、緑が train、黄土色が validation、青が test / rolling_test、灰色が split 外の補助期間を表す
- trade marker の緑線は open、赤線は close を表し、背景色の train / validation / test とは別の意味を持つ

`pair_daily_trace_manifest.csv`:

- selected event ごとの trace CSV と plot PNG のパス
- `trade_count`, `total_strategy_return_pct`, `trace_start_date`, `trace_end_date`

`selection_metrics.csv`:

- `precision_at_k`
- `ndcg_at_k`
- `spearman_score_y_corr`
- `top_k_mean_y_corr`
- `selected_mean_y_corr`
- `structural_break_rate_selected`

`trading_metrics.csv`:

- `total_return_pct`
- `sharpe_ratio`
- `max_drawdown_pct`
- `volatility`
- `hit_ratio`
- `trade_count`
- `avg_holding_days`

## 実装上の簡易化

研究計画書では Transformer Encoder、ランキング損失、DCC、Temporal GNN、RL、ISEPT との比較、attention / feature gate による解釈性分析まで含みます。この実装ではまず動く基盤として以下に絞っています。

- エンコーダは LSTM 版。
- ランキング損失は明示実装せず、`y_corr` 回帰損失を使う。
- ペア推定モデルは P&L、risk、trade direction、size を出力しない。
- Sharpe予測モデルは CAE 画像ではなく、OHLCV sequence と相関値を直接入力する。
- 売買は別モデルではなく、ISEPT と同じ VIDYAMURTHY / GATEV のband ruleが担当する。
- ペア集合選定は最大重みマッチングの厳密解ではなく、score順の greedy no-overlap。
- 比較対象のDCC / Temporal GNN / RL / ISEPT は未実装。

このため、現段階は研究計画書のプロトタイプ実装です。候補生成、再相関化ラベル、DNNスコアリング、ペア集合選定、バックテストを一通り接続することを優先しています。
