
# Competition-template

データ分析コンペ用テンプレートリポジトリ.

## How to start

コンペごとにForkして使う.
GitHubページ上部の+ボタンを押してimport repositoryを選択し、
`https://github.com/habroptilus/competition-template.git`を入力.新しいリポジトリ名はコンペの名前などを入れる.こちらはPrivateにすることができる(テンプレートリポジトリはPublicにしないとForkできないみたい)

1. `python lilac/managers/initialize.py`でセットアップする
2. コンペサイトからダウンロードしたデータを1で作成した`data`(または`luigi`)におく.
3. `main.py`の`create_sub`関数を実装する.
4. `config.yaml` のパラメータを必要に応じて変更する. 



## How to run

`luiti`直下に`train.csv`と`test.csv`がある状態で実行する.

`experiment.json`の中身を編集する.

各項目について簡単な説明:

* run : featuresとmodelの組み合わせの配列、およびstackingの方法を指定する
* features : 特徴量の集合を登録する.
* stacking : スタッキングの設定を登録する.

runの中で実行したいkeyを指定して、以下のコマンドを実行する.



`python main.py -k キー`


`-t`をつけるとパラメータのチューニングは行う.デフォルトでは予測モデルのハイパーパラメータのみチューニング対象にする.
`-fs`をつけると予測モデルに加えて特徴量選択時のfeature importanceを計算するモデルのハイパーパラメータもチューニングする.
`-th`をつけると、特徴量選択時に使用する特徴量の割合のパラメータもチューニングするようになる.(付けない場合はimportance > 0の特徴量を使用する挙動になっている.)


## How to custom

自作特徴量作成クラスは
`features/generators`の下に`FeaturesBase`を継承した特徴量作成クラスを作り、`features/dict_for_registor.py`に追加すると使えるようになる.

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




