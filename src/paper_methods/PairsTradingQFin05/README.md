# Pairs Trading QFin05

Elliott, Van der Hoek and Malcolm (2005) の状態空間モデルを使ったペアトレーディング実装です。観測 spread を隠れ状態つきの平均回帰過程として扱い、EM でパラメータを推定し、Kalman filter の one-step-ahead prediction から signal を作ります。

## 実行方法

前提データ:

- `data/raw/spy/1d.csv`
- `data/raw/qqq/1d.csv`

signal と trade record を生成します。

```bash
uv run src/paper_methods/PairsTradingQFin05/run.py
```

signal からリターン、レポート、チャートを生成します。

```bash
uv run src/gen_result/run.py
```

現在の実装では、対象ペアや interval は Python ファイル内の定数で指定しています。

```python
PAIR1 = "spy"
PAIR2 = "qqq"
INTERVAL = "1d"
WINDOW = 100
THRESHOLD = 1.0
N_EM_ITER = 150
REESTIMATE_EVERY = 20
```

## 結果ファイル構成

`run.py` は signal と trade record を `data/signals/PairsTradingQFin05/` に保存します。

```text
data/signals/PairsTradingQFin05/
├── spy_qqq_1d_signals.csv
└── spy_qqq_1d_trades.csv
```

`spy_qqq_1d_signals.csv`:

- `datetime`: signal の日時
- `spread`: `log(SPY) - beta * log(QQQ)`
- `x_pred`: Kalman filter の one-step-ahead state prediction
- `sigma_pred`: prediction variance
- `innovation`: `spread - x_pred`
- `z_score`: innovation を標準化した値
- `signal`: `1` long spread、`-1` short spread、`0` flat
- `lots`: signal がある日の基準 lot

`spy_qqq_1d_trades.csv`:

- `datetime`: open / close イベント日時
- `action`: `OPEN` または `CLOSE`
- `direction`: `LONG` または `SHORT`
- `spy_side`, `qqq_side`: 各 leg の売買方向
- `spy_lots`, `qqq_lots`: hedge ratio に基づく lot
- `z_score`, `spread`: イベント時点の状態

`src/gen_result/run.py` は評価結果を `data/results/PairsTradingQFin05/standard/kalman_em/` に保存します。

```text
data/results/PairsTradingQFin05/standard/kalman_em/
├── spy_qqq_1d_report.txt
└── spy_qqq_1d_return.png
```

`spy_qqq_1d_report.txt`:

- strategy total return
- annualised return
- Sharpe ratio
- max drawdown
- active day win rate
- benchmark buy-and-hold metrics

`spy_qqq_1d_return.png`:

- cumulative return
- drawdown
- rolling 63-day Sharpe ratio

## モデルワークフロー

1. `run.py` が `data/raw/{pair}/1d.csv` から close price を読み込む。
2. `SPY` と `QQQ` の共通日時にそろえる。
3. `spread.py` が OLS で hedge ratio `beta` を推定する。
4. `spread.py` が `log(price1) - beta * log(price2)` の spread を作る。
5. `em.py` が初期 window に対して状態空間モデルの `A, B, C^2, D^2` を EM 推定する。
6. `kalman.py` が rolling window ごとに Kalman filter を走らせる。
7. `signals.py` が one-step-ahead prediction と観測 spread の innovation を計算する。
8. innovation の z-score が `THRESHOLD` を超えたら spread の long / short signal を出す。
9. `run.py` が signal の変化を open / close trade record に変換する。
10. `src/gen_result/run.py` が signal、raw price、benchmark から日次リターンと評価レポートを作る。

## トレードルール

- `z_score > THRESHOLD`: long spread
- `z_score < -THRESHOLD`: short spread
- `abs(z_score) <= THRESHOLD`: flat

long spread は `PAIR1` を買い、`PAIR2` を hedge ratio 分だけ売ります。short spread はその逆です。

## 検証

以下の既存出力があります。

- `data/signals/PairsTradingQFin05/spy_qqq_1d_signals.csv`
- `data/signals/PairsTradingQFin05/spy_qqq_1d_trades.csv`
- `data/results/PairsTradingQFin05/standard/kalman_em/spy_qqq_1d_report.txt`
- `data/results/PairsTradingQFin05/standard/kalman_em/spy_qqq_1d_return.png`
