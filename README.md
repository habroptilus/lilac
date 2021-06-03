# Lilac

データ分析コンペ用ライブラリ.

## Install

`pip install python-lilac`でインストールする

(まだpypiに登録してないのでできない)

## SetUp

1. `lilac-init`コマンドで初期設定をする.
2. ダウンロードしたデータを1で作成した`data`(または`result/luigi`)におく.

## Run

* データセット用意
*  `experiment.json`と`config.yaml`の編集
* 自作特徴量の作成

をした後、`lilac-run`コマンドで実行する.


### データセット用意

`feature/luigi`直下に`train.csv`と`test.csv`がある状態を作る必要がある。

もともとコンペのデータセットの形式が問題なければ`features/luigi`直下におけばよく、相違がある場合はその`data`にオリジナルを置いて変換するスクリプト等を自作して`features/luigi`直下に変換したデータセットを置くようにする.


### 設定ファイルの編集
`
`settings.json`の中身を編集する.

`settings.json`の各項目について簡単な説明:

* config : デフォルトパラメータ. runやstackingのparamsで上書きできる.
* run : featuresとmodelの組み合わせの配列、およびstackingの方法を指定する
* features : 特徴量の集合を登録する.
* stacking : スタッキングの設定を登録する.


### 自作特徴量の作成

`features/generators`以下に`lilac.features.features_base.FeaturesBase`を継承して自作する.

自作した特徴量を使うために`features/__init__.py`に登録する.

### 実行

`lilac-run キー (オプション)`

キーは`experiment.json`の`run`のキーを指定する.


`-t`をつけるとパラメータのチューニングは行う.デフォルトでは予測モデルのハイパーパラメータのみチューニング対象にする.
`-fs`をつけると予測モデルに加えて特徴量選択時のfeature importanceを計算するモデルのハイパーパラメータもチューニングする.
`-th`をつけると、特徴量選択時に使用する特徴量の割合のパラメータもチューニングするようになる.(付けない場合はimportance > 0の特徴量を使用する挙動になっている.)

## lilac

* ensemble : アンサンブル、スタッキング関連
* evaluators : 評価指標を計算する
* experiment : 実験を実行する
* features : 特徴量生成
* managers : セットアップなど
* models : 予測モデル
* preprocessors : 特徴量生成に用いるクラス、関数など
* selectors : 特徴量選択を行う.
* tasks : luigiタスク
* trainers : モデルの学習を行う.
* tuner : ハイパーパラメータチューニング
* utils : その他
* validators : 交差検証のためのコード




