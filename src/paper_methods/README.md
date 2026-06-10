# Paper Methods

このディレクトリは、論文ごとのペアトレーディング手法を実装する場所です。各論文フォルダには、その論文単位で次の三つを説明する `README.md` を置きます。

1. 実行方法
2. 結果ファイルの構成
3. モデルとトレードのワークフロー

## 論文別README

- `PairsTradingQFin05/README.md`
  - Elliott, Van der Hoek and Malcolm (2005) の状態空間モデル、EM、Kalman filter によるペアトレーディングです。
  - 主な出力先は `data/signals/PairsTradingQFin05/standard/kalman_em/` と `data/results/PairsTradingQFin05/standard/kalman_em/` です。

- `ml/Enhancing a Pairs Trading strategy with the application of Machine Learning/README.md`
  - Sarmento and Horta (2020) の PCA / OPTICS によるペア選定と、spread 予測モデルによるトレード改善です。
  - 主な出力先は `data/results/{output_name}/{rule_or_model}/{model_variant}/` と `data/signals/{output_name}/{rule_or_model}/{model_variant}/` です。
  - LSTM / encoder-decoder は Modal GPU でも学習・推論できます。

## READMEの標準構成

新しい論文実装を追加するときは、論文フォルダ直下に `README.md` を作り、最低限次の見出しを入れます。

- `実行方法`: 必要なデータ、依存関係、基本コマンド、代表的なオプション
- `結果ファイル構成`: `data/results`、`data/signals`、外部保存先があればその構造と各ファイルの意味
- `モデルワークフロー`: データ入力から特徴量、学習、推論、signal、trade、評価までの手順
- `主要設定`: 銘柄、期間、window、threshold、model backend など、再現性に効く設定
- `検証`: そのREADMEを書いた時点で通したコマンドやスモークテスト
