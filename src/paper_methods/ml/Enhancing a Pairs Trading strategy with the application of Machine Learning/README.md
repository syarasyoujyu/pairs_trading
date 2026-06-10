# Enhancing a Pairs Trading Strategy with Machine Learning

Sarmento and Horta (2020) の手法を、ローカルの `data/raw/{symbol}/{interval}.csv` に対して動かす実装です。論文は大きく二つの改善を提案しています。

1. PCA と OPTICS でペア候補の探索空間を絞る
2. spread の将来変化を予測し、長い逆行期間を避ける

このディレクトリでは、各処理を小さなモジュールに分けています。

## データ収集

```bash
uv run src/data/gen_data.py --period 10y --interval 1d --chunk-size 40
```

`src/data/gen_data.py` は State Street の SPY holdings xlsx を取得し、そこに含まれる USD 建ての通常株ティッカーを yfinance 形式へ変換します。

出力:

- `data/universe/spy_holdings.csv`: SPY の構成銘柄一覧
- `data/raw/{symbol}/1d.csv`: 各銘柄と SPY の調整後 OHLCV

`spy_holdings.csv` の意味:

- `SPY` は SPDR S&P 500 ETF Trust です。
- `spy_holdings.csv` は SPY の価格データではなく、SPY が保有している銘柄一覧です。
- この実装では、この銘柄一覧を「ペア候補を探す universe」として使います。
- `Ticker` は公式 holdings xlsx の表記、`yfinance_symbol` は yfinance 取得用、`storage_symbol` は `data/raw/{symbol}/` 保存用です。
- `Weight` は SPY 内での保有比率ですが、今のペアトレードでは weighting には使わず、銘柄 universe の確認情報として保存しています。

価格を再取得せずに構成銘柄CSVだけ更新したい場合は、次のように `--limit 0` を使います。

```bash
uv run src/data/gen_data.py --period 10y --interval 1d --limit 0
```

## バックテスト実行

```bash
uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py"
```

`run.py` の主な設定です。

```python
INTERVAL = "1d"
FORECAST_MODEL = "arma"  # arma, rolling_ar, lstm, encoder_decoder
NEURAL_BACKEND = "modal"  # modal, local
FORMATION_BARS = 756
TRADING_BARS = 252
MAX_SELECTED_PAIRS = 20
ARMA_CONFIG = best_paper_arma_config()  # p=8, q=3, h=1
LSTM_CONFIG = best_paper_lstm_config()  # in=24, hl=1, hn=50, h=1
ENCODER_DECODER_CONFIG = best_paper_encoder_decoder_config()  # in=24, en=15, dn=15, h=2
```

`run.py` は `data/universe/spy_holdings.csv` を読み、SPY 構成銘柄と SPY ベンチマークを `data/raw` から読み込みます。SPY ベンチマークが存在する最新日を基準に直近 `FORMATION_BARS + TRADING_BARS` 本を使い、欠損の多い銘柄を落としてから、形成期間でペアを選定し、取引期間で標準モデルと予測モデルを比較します。

LSTM 系を使う場合は `FORECAST_MODEL` を `lstm` または `encoder_decoder` に変更します。必要な `scikit-learn`、`statsmodels`、`tensorflow`、`openpyxl`、`modal` は `uv add` で依存に入れているため、実行は `uv run` 経由にそろえます。

CLI から設定を上書きできます。

```bash
uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py" --forecast-model arma
uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py" --forecast-model rolling_ar
uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py" --forecast-model lstm --neural-backend local
uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py" --forecast-model lstm --neural-backend modal
uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py" --forecast-model encoder_decoder --neural-backend modal
uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py" --forecast-model lstm --neural-backend modal --max-selected-pairs 1 --output-name EnhancingPairsTradingML_lstm_modal_smoke
```

## Modal GPU 実行

Modal を使う場合は、事前にローカルで Modal にログインしておきます。`.env` に `MODAL_TOKEN_ID` と `MODAL_TOKEN_SECRET` がある場合は、値を表示せずに次のコマンドで設定できます。

```bash
set -a; source .env; set +a
uv run modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET" --activate --verify
```

ブラウザで認証する場合は次のコマンドを使います。

```bash
uv run modal setup
```

その後、次のコマンドで LSTM の学習と推論を Modal GPU 上で実行します。

```bash
uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py" --forecast-model lstm --neural-backend modal
```

既存の結果を上書きせずに疎通確認する場合は、次のように出力名を変えます。

```bash
uv run "src/paper_methods/ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/run.py" --forecast-model lstm --neural-backend modal --max-selected-pairs 1 --output-name EnhancingPairsTradingML_lstm_modal_smoke
```

処理の流れ:

1. ローカルでペア選定まで行う。
2. 採用ペアごとの spread を training / validation / prediction payload に変換する。
3. `modal_execution/client.py` が ephemeral Modal app を起動する。
4. `modal_execution/app.py` の `train_and_predict_spread` が GPU 関数として実行される。
5. Modal 側で Keras LSTM または encoder-decoder を学習する。
6. 同じ Modal 関数内で validation 期間と trading 期間の予測を実行する。
7. 学習済みモデルを `enhancing-pairs-trading-ml-models` Volume に `.keras` 形式で保存する。
8. 予測結果をローカルへ戻し、既存の予測ベーストレード手順で signal と return を作る。

Modal 側のデフォルト GPU は `L4`、フォールバックは `T4` です。変更する場合は `modal_execution/app.py` の `GPU_FALLBACKS` を編集します。保存済みモデルで後から推論する関数として `predict_saved_spread` も用意しています。

## 結果ファイル構成

`run.py` は `--output-name` をトップ階層にし、forecast model と backend で結果と signal を細分化して保存します。デフォルトは `EnhancingPairsTradingML` です。

```text
data/results/{output_name}/{forecast_model}/
data/results/{output_name}/{forecast_model}/{neural_backend}/
data/signals/{output_name}/{forecast_model}/
data/signals/{output_name}/{forecast_model}/{neural_backend}/
```

`lstm` と `encoder_decoder` は backend 比較も含むため `{neural_backend}` 階層を使います。`arma` と `rolling_ar` は backend を使わないため `{forecast_model}` までです。

```text
data/results/EnhancingPairsTradingML/lstm/modal/
├── cluster_labels.csv
├── candidate_diagnostics.csv
├── selected_pairs.csv
├── cluster_plot_manifest.csv
├── pair_summary.csv
├── pair_trading_metrics.csv
├── portfolio_comparison.csv
├── paper_trading_metrics.csv
├── forecast_error_metrics.csv
├── forecast_error_summary.csv
├── portfolio_daily_returns.csv
├── pair_daily_traces.csv
├── pair_daily_trace_manifest.csv
├── pair_daily_traces/
├── pair_diagnostic_plots/
└── plots/
    ├── clusters/
    │   └── cluster_{cluster_id}_normalized_returns.png
    └── pairs/
        └── {pair}/
            ├── {pair}_price_movement.png
            ├── {pair}_standard_trade_timing.png
            └── {pair}_{forecast_model}_trade_timing.png

data/signals/EnhancingPairsTradingML/lstm/modal/
├── {pair}_standard_signals.csv
├── {pair}_standard_trades.csv
├── {pair}_{forecast_model}_signals.csv
└── {pair}_{forecast_model}_trades.csv
```

代表例:

```text
data/results/EnhancingPairsTradingML/arma/
data/signals/EnhancingPairsTradingML/arma/

data/results/EnhancingPairsTradingML/lstm/modal/
data/signals/EnhancingPairsTradingML/lstm/modal/
```

`cluster_labels.csv`:

- 銘柄ごとの OPTICS cluster label
- ペア候補を作る前のクラスタリング結果確認に使う

`candidate_diagnostics.csv`:

- PCA / OPTICS で作った全ペア候補の診断結果
- `adf_t_stat`, `adf_p_value`, `hurst`, `half_life_bars`, `crossings_per_year`
- `is_selected` と `rejection_reason` で採用 / 不採用理由を見る

`selected_pairs.csv`:

- 採用されたペアだけを保存する
- `asset_y`, `asset_x`, `hedge_ratio`, `intercept` が実トレードで使う spread 定義になる

`cluster_plot_manifest.csv`:

- クラスタごとの画像ファイルと構成銘柄を保存する
- `cluster`: クラスタID
- `member_count`: クラスタ内の銘柄数
- `members`: クラスタ内銘柄
- `plot_path`: `plots/clusters/cluster_{cluster_id}_normalized_returns.png` へのパス

`pair_summary.csv`:

- ペアごとの `standard_threshold` と `forecast_{model}` の成績
- total return、annualized return、Sharpe、max drawdown、active days を比較する

`pair_trading_metrics.csv`:

- 論文表で使われた取引指標をペア単位で保存する
- `SR`, `ROI_pct`, `MDD_pct`, `days_of_portfolio_decline`
- `total_pairs`, `profitable_pairs`, `profitable_pairs_pct`
- `total_trades`, `profitable_trades`, `unprofitable_trades`
- `portfolio_volatility_pct`, `annualization_factor`

`portfolio_comparison.csv`:

- equal-weight portfolio と SPY buy-and-hold の比較
- `role=strategy` の行が実際のペアトレード戦略
- `role=benchmark` の行は比較用の SPY buy-and-hold
- strategy は `standard_threshold` と `forecast_{model}` の二つ
- `SPY buy-and-hold` はモデルではなく、市場ベンチマーク
- 論文手法の最終比較を見る主要ファイル

`paper_trading_metrics.csv`:

- 論文の Table 3 / Table 5 / Table 6 に対応するポートフォリオ単位の取引指標
- `SR`: annualized Sharpe Ratio
- `ROI_pct`: Return on Investment。初期資本を 1 とした total return
- `MDD_pct`: Maximum Drawdown。論文表に合わせて正のパーセントで保存
- `days_of_portfolio_decline`: portfolio daily return が負だった日数
- `total_pairs`: portfolio に入ったペア数
- `profitable_pairs`: total return が正のペア数
- `profitable_pairs_pct`: profitable pairs の割合
- `total_trades`: non-flat holding episode の数
- `profitable_trades`: trade return が正の trade 数
- `unprofitable_trades`: trade return が 0 以下の trade 数
- `portfolio_volatility_pct`: portfolio daily return の標準偏差
- `annualization_factor`: daily data では `sqrt(252)`

`forecast_error_metrics.csv`:

- 論文の Table 4 に対応する予測誤差指標
- ペアごとに validation と test の両方を保存する
- `ar_order`, `ma_order`, `input_length`, `hidden_layers`, `hidden_units`, `encoder_units`, `decoder_units`: その実行で使った予測モデル設定
- `mse`, `rmse`, `mae`: raw spread scale の予測誤差
- `mse_e03`, `rmse_e02`, `mae_e02`: 論文表の `MSE (E-03)`, `RMSE (E-02)`, `MAE (E-02)` に合わせた表示用スケール

`forecast_error_summary.csv`:

- `forecast_error_metrics.csv` を `model`, `period`, `horizon` ごとに平均したもの
- 論文 Table 4 は複数spreadの平均MSE/RMSE/MAEで比較しているため、このsummaryがTable 4に近い見方になります
- `pair_count`: 平均に使ったペア数

`portfolio_daily_returns.csv`:

- strategy と benchmark の日次リターン
- `standard_threshold`、`forecast_{model}`、`benchmark_spy_buy_hold` の列を持つ
- 外部で追加分析、plot、統計検定をしたい場合に使う

`pair_daily_traces.csv`:

- selected pair ごとの日次トレースを縦持ちで保存する
- `model_split_period`: `train`, `validation`, `test`
- `event_phase`: `formation_train`, `formation_validation`, `trading_test`
- `raw_pair_daily_return`: signal を掛ける前の equal-dollar pair return
- `standard_daily_return`, `standard_cumulative_return`: standard threshold の日次/累積return
- `forecast_daily_return`, `forecast_cumulative_return`: `forecast_{model}` の日次/累積return
- `standard_signal`, `forecast_signal`: `1` は long spread、`-1` は short spread、`0` は flat
- `standard_trade_action`, `forecast_trade_action`: open / close / switch のタイミング
- `standard_long_entry`, `standard_short_entry`, `standard_exit`: standard threshold で使う指標
- `forecast_predicted_spread`, `forecast_predicted_change_pct`, `forecast_long_entry_pct`, `forecast_short_entry_pct`: forecast trading で使う指標

`pair_daily_traces/{pair}.csv`:

- `pair_daily_traces.csv` を selected pair ごとに分割したCSV
- ペア単位で train / validation / test と売買を確認する時に使う

`pair_diagnostic_plots/{pair}.png`:

- 上段: 2銘柄の相対価格推移
- 2段目: spread と standard threshold
- 3段目: forecast の予測変化率または予測spreadと threshold
- 4段目: standard / forecast の signal
- 下段: standard / forecast の日次returnと累積return
- 背景色で train / validation / test を区別する

`pair_daily_trace_manifest.csv`:

- selected pair ごとの trace CSV と plot PNG のパス
- `standard_total_return_pct`, `forecast_total_return_pct`

`plots/clusters/cluster_{cluster_id}_normalized_returns.png`:

- クラスタリング入力に使った `normalized_returns()` の値動き
- 各日の `R_i,t = (P_i,t - P_i,t-1) / P_i,t-1` を%表示する
- 薄い線がクラスタ内の各銘柄、太い線がクラスタ平均
- 累積リターンや価格指数ではなく、各日で相対的にどれだけ動いたかを描く

`plots/pairs/{pair}/{pair}_price_movement.png`:

- ペアに含まれる2銘柄の価格を初日比の%変化にそろえた図
- training / validation / trading の全期間を含む
- 背景色は training、validation、trading の区間を表す
- 茶色の点線は training / validation の境目、黒の点線は validation / trading の境目
- 下段の timing panel に standard と forecast の long / short entry を表示する
- 上向き三角が long entry、下向き三角が short entry

`plots/pairs/{pair}/{pair}_standard_trade_timing.png` と `plots/pairs/{pair}/{pair}_{forecast_model}_trade_timing.png`:

- spread 上に long / short / close のタイミングを重ねた図
- 緑の背景と上向き三角が long、赤の背景と下向き三角が short、黒の x が close
- 予測モデル側は `predicted_spread` も点線で表示する

`SPY buy-and-hold` と `benchmark_spy_buy_hold` の意味:

- これは LSTM や threshold model ではありません。
- SPY を取引期間の最初から最後まで買って持ち続けた場合の日次リターンです。
- ペアトレード戦略が市場全体の単純保有に対してどうだったかを見るための benchmark です。
- `portfolio_comparison.csv` では `role=benchmark`、`portfolio_daily_returns.csv` では `benchmark_spy_buy_hold` として保存します。

## 論文指標との対応

この実装で測れる論文指標は次の通りです。

| 論文中の指標 | 出力CSV | 列名 |
| --- | --- | --- |
| Sharpe Ratio | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `SR` |
| Return on Investment | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `ROI_pct` |
| Maximum Drawdown | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `MDD_pct` |
| Days of portfolio decline | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `days_of_portfolio_decline` |
| Total pairs | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `total_pairs` |
| Profitable pairs | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `profitable_pairs` |
| Profitable pairs (%) | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `profitable_pairs_pct` |
| Total trades | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `total_trades` |
| Profitable trades | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `profitable_trades` |
| Unprofitable trades | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `unprofitable_trades` |
| Trades (Positive-Negative) | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `profitable_trades`, `unprofitable_trades` |
| Portfolio volatility | `paper_trading_metrics.csv`, `pair_trading_metrics.csv` | `portfolio_volatility_pct` |
| MSE | `forecast_error_metrics.csv`, `forecast_error_summary.csv` | `mse`, `mse_e03` |
| RMSE | `forecast_error_metrics.csv`, `forecast_error_summary.csv` | `rmse`, `rmse_e02` |
| MAE | `forecast_error_metrics.csv`, `forecast_error_summary.csv` | `mae`, `mae_e02` |

注意:

- 論文は5分足ETFデータと取引コストを前提にしています。
- このリポジトリの現在のSPY universe実行は日足データを使います。
- `annualization_factor` は日足前提の `sqrt(252)` です。
- 取引コスト列はまだ分けていません。現在の指標は、この実装で計算した日次リターンに基づきます。

`data/signals/{output_name}/{forecast_model}/{backend}/{pair}_{model}_signals.csv`:

- ペア別・モデル別の signal
- spread、予測値、position、entry / exit 判定を追うためのファイル

`data/signals/{output_name}/{forecast_model}/{backend}/{pair}_{model}_trades.csv`:

- ペア別・モデル別の open / close trade record
- leg ごとの方向と hedge ratio を確認するためのファイル

Modal で LSTM / encoder-decoder を実行した場合、学習済み Keras モデルは Modal Volume に保存されます。

```text
Modal Volume: enhancing-pairs-trading-ml-models
└── {pair}_{model_kind}_{formation_start}_{formation_end}/
    ├── model.keras
    └── metadata.json
```

## モデル全体ワークフロー

1. `src/data/gen_data.py` が SPY holdings と価格データを用意する。
2. `run.py` が価格を読み込み、形成期間と取引期間へ分割する。
3. `pair_selection/` が PCA と OPTICS でペア候補を絞る。
4. `pair_selection/diagnostics.py` が cointegration、Hurst exponent、half-life、mean crossing を検査する。
5. 採用ペアごとに `log(Y) - beta * log(X) - intercept` の spread を作る。
6. 標準モデルは formation spread の平均と標準偏差から entry / exit threshold を決める。
7. 予測モデルは training / validation / trading に分け、spread の将来値を rolling AR、LSTM、encoder-decoder のいずれかで予測する。
8. LSTM / encoder-decoder で `--neural-backend modal` を指定した場合、学習と推論は Modal GPU 上で実行する。
9. validation 期間で予測変化率 threshold を選び、trading 期間で signal を生成する。
10. `execution/` がペア別リターン、equal-weight portfolio、SPY buy-and-hold 比較を保存する。

## ペア選定の手順

1. `data_loading/prices.py` が `data/universe/spy_holdings.csv` の銘柄に対応する close price を読み込む。
2. 直近 `FORMATION_BARS + TRADING_BARS` 本だけを取り出し、十分な履歴と低い欠損率を満たす銘柄に絞る。
3. 形成期間と取引期間に分ける。
4. `data_loading/prices.py` が各銘柄の close price を読み込み、形成期間の共通日時だけにそろえる。
5. `pair_selection/features.py` が価格から正規化リターン `Ri,t = (Pi,t - Pi,t-1) / Pi,t-1` を作る。
6. 同じく `pair_selection/features.py` がリターン行列を PCA にかけ、各銘柄を少数次元の特徴量で表す。
7. `pair_selection/clustering.py` が PCA 特徴量に OPTICS を適用し、同じクラスタ内の銘柄だけをペア候補にする。
8. クラスタ内ペアが多すぎる場合は PCA 空間で近い順に候補を制限する。
9. `pair_selection/selection.py` が候補ごとに `pair_selection/diagnostics.py` の検査を実行する。
10. 次の四条件をすべて満たすペアだけを採用する。

採用条件:

- Engle-Granger 検定で cointegrated と判定される
- spread の Hurst exponent が `0.5` 未満
- half-life が 1日以上、1年以下
- spread が平均を年 12 回以上横切る

出力:

- `cluster_labels.csv`: 銘柄ごとのクラスタ
- `candidate_diagnostics.csv`: 全候補の診断値と落選理由
- `selected_pairs.csv`: 採用ペア

## 標準トレード手順

標準モデルは Gatev et al. 型のしきい値モデルです。`trading/thresholds.py` と `trading/signals.py` が担当します。

1. formation 期間の spread から平均 `mu` と標準偏差 `sigma` を計算する。
2. long entry を `mu - 2 * sigma`、short entry を `mu + 2 * sigma`、exit を `mu` にする。
3. trading 期間で spread を監視する。
4. spread が long entry 以下なら spread を long する。
5. spread が short entry 以上なら spread を short する。
6. long は spread が平均以上に戻ったら閉じる。
7. short は spread が平均以下に戻ったら閉じる。

spread の long は `Y を買い、X を beta 分だけ売る` 取引です。spread の short はその逆です。

## 予測ベースのトレード手順

予測ベースモデルは、spread が平均からどれだけ離れたかではなく、予測される spread の変化率で売買します。

1. formation 期間を training と validation に分ける。
2. training で spread 予測モデルを学習する。
3. training 側の spread 変化率分布から二つのしきい値候補を作る。
4. validation で quintile しきい値と decile しきい値を試す。
5. validation return が高いしきい値セットを採用する。
6. trading 期間で `S*(t+h) - S(t)` の予測変化率を計算する。
7. 予測変化率が long threshold 以上なら spread を long する。
8. 予測変化率が short threshold 以下なら spread を short する。
9. 保有中のポジションは予測方向が反転したら閉じる。

予測変化率:

```text
D(t+h) = (S*(t+h) - S(t)) / S(t) * 100
```

`h=1` は ARMA/LSTM の単一ステップ予測、`h=2` は encoder-decoder の二ステップ予測に使います。

## モデル学習の手順

### ARMA

論文Table 4で強調されているARMA設定は `p=8, q=3` です。この実装では `forecasting_models/forecasting.py` の `rolling_arma_forecast` が担当します。内部では statsmodels の `ARIMA(order=(p, 0, q))` API を使うため名前はARIMAですが、差分次数 `d=0` なので論文のARMAです。

1. training spread で ARMA(p, q) を一度推定する。
2. validation / trading の各時刻で、その時刻の実測 spread をモデル状態へ追加する。
3. 追加後の状態から `S*(t+horizon)` を予測する。
4. デフォルトは `ARMAForecastConfig(ar_order=8, ma_order=3, horizon=1)`。

`rolling_ar` は軽量な比較用です。論文のARMAそのものではないため、デフォルトにはしていません。

### LSTM

`forecasting_models/neural_training.py`、`forecasting_models/neural_models.py`、`forecasting_models/neural_prediction.py` が担当します。

1. `forecasting_models/scaling.py` が training spread の平均と標準偏差で標準化する。
2. `forecasting_models/sequences.py` が `input_length` 本の過去 spread から `horizon` 本先を当てる教師データを作る。
3. `forecasting_models/neural_models.py` が LSTM + Dense の Keras モデルを作る。
4. `forecasting_models/neural_training.py` が MSE loss で学習する。
5. validation がある場合は early stopping を使い、最良 weight に戻す。
6. `forecasting_models/neural_prediction.py` が現在時刻 `t` に対して `S*(t+horizon)` を返す。

代表設定は `forecasting_models/paper_model_configs.py` に置いています。

```python
LSTMForecastConfig(input_length=24, hidden_layers=1, hidden_units=50, horizon=1)
```

論文Table 4の表記では `in=24, hl=1, hn=50` です。`in=24` は「入力系列長が24」という意味で、24層のLSTMという意味ではありません。

LSTM の入力:

- この実装の LSTM は、全ペアで共有される単一モデルではありません。
- 1つのペア、つまり2つの銘柄/ETFから作られる1本の spread 時系列に対して、1つの LSTM を学習します。
- たとえば `googl_goog`、`adp_payx`、`eog_xom` が採用された場合、それぞれ別々の LSTM が用意されます。
- `LSTM_CONFIG` はモデル構造と学習条件のテンプレートであり、学習済み weight を共有するという意味ではありません。
- `MAX_SELECTED_PAIRS=20` で LSTM を使う場合、最大20個の LSTM が順番に学習されます。
- LSTM は株価そのものや OHLCV を直接入力しません。
- 入力するのは、採用ペアごとに作った spread の時系列です。
- spread は `S_t = log(P_y,t) - beta * log(P_x,t) - intercept` です。
- training spread の平均と標準偏差で標準化してから使います。
- `input_length=24` の場合、過去 24 本の標準化 spread を 1 サンプルにします。
- Keras に渡す `x_train` の shape は `(samples, input_length, 1)` です。

LSTM の教師データ:

- `horizon=1` の場合、過去 24 本から 1 本先の spread を当てます。
- `y_train` の shape は `(samples, horizon)` です。
- 例として `input_length=24, horizon=1` なら、`[S_t-23, ..., S_t]` から `S_t+1` を学習します。

LSTM の出力:

- モデルの raw output は標準化された将来 spread です。
- `neural_prediction.py` が元の spread scale に戻し、`predicted_spread` として返します。
- signal CSV では `spread`、`predicted_spread`、`predicted_change_pct` が確認できます。
- `predicted_change_pct = (predicted_spread - current_spread) / current_spread * 100` です。
- トレードでは `predicted_spread` 自体ではなく、`predicted_change_pct` が long / short threshold を超えるかを使います。

LSTM 実行時に出る主なファイル:

- `data/signals/{output_name}/lstm/{backend}/{pair}_lstm_signals.csv`
- `data/signals/{output_name}/lstm/{backend}/{pair}_lstm_trades.csv`
- `data/results/{output_name}/lstm/{backend}/pair_summary.csv`
- `data/results/{output_name}/lstm/{backend}/portfolio_comparison.csv`

Modal を使う場合の保存先:

- 学習済み Keras モデルは `enhancing-pairs-trading-ml-models` Volume に保存します。
- 保存名は `{pair}_lstm_{formation_start}_{formation_end}` です。
- metadata には `model_scope=per_pair_spread_forecaster` を保存します。
- 保存ディレクトリがペアごとに分かれるので、後から保存済みモデルで推論するときもペア別モデルを指定します。

### LSTM Encoder-Decoder

encoder-decoder は複数ステップ先の spread をまとめて予測するモデルです。

1. encoder LSTM が過去系列を固定長ベクトルに圧縮する。
2. `RepeatVector(horizon)` で decoder へ渡す系列を作る。
3. decoder LSTM が `horizon` 本ぶんの出力系列を作る。
4. 最後のステップを `S*(t+horizon)` としてトレード判定に使う。

代表設定:

```python
EncoderDecoderForecastConfig(input_length=24, encoder_units=15, decoder_units=15, horizon=2)
```

論文Table 4の表記では `in=24, en=15, dn=15` です。Section 5.2で encoder-decoder は output length two とされているため、この実装のデフォルトは `horizon=2` にしています。

## 予測モデルの比較

`forecasting_models/forecast_experiments.py` の `compare_validation_forecasts` で、validation 区間の予測誤差を比較できます。

```python
from forecasting_models.forecast_experiments import compare_validation_forecasts
from forecasting_models.paper_model_configs import best_paper_arma_config, best_paper_lstm_config

report = compare_validation_forecasts(
    spread,
    validation_start=600,
    formation_end=756,
    arma_config=best_paper_arma_config(),
    lstm_config=best_paper_lstm_config(),
)
```

返り値には `model`, `horizon`, `mse`, `rmse`, `mae` が入ります。

## ファイル構成

- `common/`: 共有 dataclass と設定コンテナ
- `data_loading/`: ローカル価格データの読み込み
- `pair_selection/`: PCA、OPTICS、cointegration 診断、ペア選定
- `forecasting_models/`: naive / rolling AR / ARMA / LSTM / encoder-decoder と予測評価
- `modal_execution/`: Modal GPU 上のニューラルモデル学習・推論
- `trading/`: 標準/予測ベースのしきい値、signal、trade record、validation
- `execution/`: ポートフォリオ日次リターン、SPY比較、CSV保存
- `run.py`: 各工程をつなぐ実行入口
