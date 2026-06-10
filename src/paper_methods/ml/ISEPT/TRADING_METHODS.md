# Trading Methods: VIDYAMURTHY and GATEV

このファイルは、ISEPTで選定されたペアをどの売買ルールで執行するかを説明します。
README.mdは実行入口、このファイルは売買手法の仕様メモです。

## 前提

ISEPT本体のペア選定は、どちらの売買ルールでも共通です。

1. OHLCをcandlestick画像にする。
2. CAEで銘柄ごとのlatent vectorを作る。
3. 2銘柄のlatent vectorを連結し、PCA後にMLPへ入れる。
4. MLPが予測Sharpe ratioを出す。
5. 予測Sharpe上位ペアを売買対象にする。

`--trade-rule vidyamurthy` と `--trade-rule gatev` の違いは、主に次の2点です。

- MLPの教師ラベルになる `pair_feedback_labels.csv` の実現Sharpeを、どの売買ルールで計算するか。
- 選定後の `signals.csv` / `trades.csv` / `portfolio_daily_returns.csv` を、どの売買ルールで作るか。

つまり、同じ銘柄 universe でも、`trade_rule` を変えると学習ラベルも最終トレードも変わります。比較する場合は、同じ `--max-assets`、`--months`、seed、モデル設定で、`--output-name` だけ分けて実行します。

## 共通のSpread推定

この実装では、両方の執行ルールで formation window のlog priceからspreadを推定します。

```text
log(P_i,t) = intercept + gamma * log(P_j,t) + error_t
spread_t = log(P_i,t) - intercept - gamma * log(P_j,t)
```

formation window:

- デフォルトは選定月末までの `252` 営業日。
- `spread_mean` と `spread_std` はformation window上のspreadから計算します。

trading window:

- デフォルトは選定月の翌月から `6` か月。
- `--trading-horizon-months` で変更できます。

positionの意味:

- `signal = 1`: long spread。実装上は asset_i 側を買い、asset_j 側を `gamma` でヘッジする方向です。
- `signal = -1`: short spread。実装上は asset_i 側を売り、asset_j 側を `gamma` でヘッジする方向です。
- returnは `1 + abs(gamma)` で正規化し、signal変化時にone-way transaction costを引きます。

## VIDYAMURTHY

### 考え方

Vidyamurthyの考え方は、cointegrationで得られるmean-revertingなspreadを売買対象にするものです。spreadが長期均衡から十分に離れたらポジションを持ち、spreadの振動を利用して利益を狙います。

`references/basic/Quantitative Methods and Analysis.pdf` では、spreadの十分な乖離を `Delta` として扱い、Gaussian white-noiseに近いspreadのband designでは `0.75 * sigma` が代表的な閾値として示されています。ISEPT論文でも、`ISEPT + VIDYAMURTHY` は `0.75σ` bandを使う構成です。

### 売買ルール

```text
delta = vidyamurthy_threshold_sigma * spread_std
lower_band = spread_mean - delta
upper_band = spread_mean + delta
```

デフォルト:

```text
vidyamurthy_threshold_sigma = 0.75
```

entry:

- flat状態で `spread <= lower_band` なら long spread。
- flat状態で `spread >= upper_band` なら short spread。

exit:

- long spread中に `spread >= upper_band` へ到達したらclose。
- short spread中に `spread <= lower_band` へ到達したらclose。
- trading window終了時点で未決済なら、その時点までのreturnで評価します。

### 実行例

```bash
uv run src/paper_methods/ml/ISEPT/run.py \
  --trade-rule vidyamurthy \
  --vidyamurthy-threshold-sigma 0.75 \
  --output-name ISEPT_vidyamurthy
```

### 出力の見方

`signals.csv` には次のVidyamurthy用カラムが出ます。

- `spread`
- `signal`
- `lower_band`
- `upper_band`
- `delta`
- `gamma`
- `intercept`
- `spread_mean`
- `spread_std`
- `trade_rule`

`pair_feedback_labels.csv` の `trade_rule` は `vidyamurthy` になり、MLPはVidyamurthy執行での実現Sharpeを学習します。

## GATEV

### 考え方

Gatev型のペアトレードは、formation windowでペアのspread分布を見て、spreadが大きく開いたときにmean reversionを狙います。ISEPT論文における `ISEPT + GATEV` は、ペアの選定だけをISEPTのCAE-MLPに置き換え、売買執行はGATEV系の閾値ルールを使う構成です。

純粋なGATEV baselineでは、formation windowの累積リターンを正規化し、2銘柄間の距離が小さいペアを選びます。この実装の `--trade-rule gatev` は、純粋なGATEVペア選定ではなく、ISEPTで選ばれたペアにGATEV執行を適用します。

### 売買ルール

この実装では、ISEPT論文のstrategy設定に合わせて、entryを `2σ`、exitを `1σ` としています。

```text
long_entry = spread_mean - 2.0 * spread_std
short_entry = spread_mean + 2.0 * spread_std
long_exit = spread_mean - 1.0 * spread_std
short_exit = spread_mean + 1.0 * spread_std
```

entry:

- flat状態で `spread <= long_entry` なら long spread。
- flat状態で `spread >= short_entry` なら short spread。

exit:

- long spread中に `spread >= long_exit` へ戻ったらclose。
- short spread中に `spread <= short_exit` へ戻ったらclose。
- trading window終了時点で未決済なら、その時点までのreturnで評価します。

### 実行例

```bash
uv run src/paper_methods/ml/ISEPT/run.py \
  --trade-rule gatev \
  --entry-sigma 2.0 \
  --exit-sigma 1.0 \
  --output-name ISEPT_gatev
```

### 出力の見方

`signals.csv` には次のGATEV用カラムが出ます。

- `spread`
- `signal`
- `long_entry`
- `short_entry`
- `long_exit`
- `short_exit`
- `beta`
- `intercept`
- `spread_mean`
- `spread_std`
- `trade_rule`

`pair_feedback_labels.csv` の `trade_rule` は `gatev` になり、MLPはGATEV執行での実現Sharpeを学習します。

## 使い分け

VIDYAMURTHY:

- `0.75σ` の薄いbandを使うため、GATEVより取引が発生しやすい傾向があります。
- 反対側bandまで待つため、holding期間が長くなることがあります。
- cointegration spreadの振動を広く取りにいく設計です。

GATEV:

- `2σ` entryなので、より大きな乖離だけを狙います。
- `1σ` exitなので、mean付近へ戻る前に利確・損切りが起きやすいです。
- 取引頻度はVIDYAMURTHYより低くなりやすい一方、entry時点の乖離は大きくなります。

## 実装対応表

```text
common/config.py          TradingConfig
run.py                    --trade-rule, --entry-sigma, --exit-sigma, --vidyamurthy-threshold-sigma
trading/strategy.py       trade_rule dispatcher
trading/vidyamurthy.py    VIDYAMURTHY signals and simulation
trading/gatev.py          GATEV signals and simulation
labels/pair_labels.py     trade_rule別の実現Sharpeラベル作成
trading/simulation.py     trade_rule別の最終トレード
```

## 注意

- `VIDYAMURTHY` と `GATEV` の名前は、ここではISEPTで選定されたペアに適用する執行ルールを指します。
- 純粋なVIDYAMURTHY baselineの相関フィルタ + Engle-Granger cointegration選定や、純粋なGATEV baselineのminimum-distance選定は、この `--trade-rule` 切り替えでは実行しません。
- 純粋baselineとISEPT拡張版を比較したい場合は、別のpair selection moduleとして実装する必要があります。
