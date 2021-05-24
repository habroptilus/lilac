from sklearn.ensemble import RandomForestRegressor
from lilac.preprocessors.utils.encoders import OrdinalEncoder
import numpy as np


class _RfBase:
    """Randomforestベースモデル.

    :カテゴリ: ラベルエンコーディング(欠損も1カテゴリになる)
    :数値: 欠損列を-1e8で補完(木系ならこれでよし)
    :inf,-inf : 置換
    """
    null_value = -1e8
    inf_value = 1e9

    def __init__(self, criterion, seed):
        self.seed = seed
        self.criterion = criterion
        self.encoder = OrdinalEncoder()

    def fit(self, train_x, train_y):
        train_x = self.encoder.fit_transform(train_x)
        train_x = train_x.fillna(self.null_value)
        train_x = train_x.replace([np.inf], self.inf_value)
        train_x = train_x.replace([-np.inf], -self.inf_value)

        self.model = RandomForestRegressor(
            criterion=self.criterion, random_state=self.seed)

        self.model.fit(train_x, train_y)
        return self

    def return_flag(self):
        return "rf"


class _RfRegressor(_RfBase):
    """ランダムフォレスト回帰モデル."""

    def predict(self, test):
        test = self.encoder.transform(test)
        test = test.fillna(self.null_value)
        test = test.replace([np.inf], self.inf_value)
        test = test.replace([-np.inf], -self.inf_value)

        return self.model.predict(test)

    def return_flag(self):
        return f"{super().return_flag()}_reg"


class _RfRmsleRegressor(_RfRegressor):
    """Randomforest回帰モデル"""

    def __init__(self, seed):
        super().__init__("mse", seed)

    def fit(self, train_x, train_y):
        # yをlog変換
        train_y = self._pre_process_y(train_y)
        # あとはもとのと一緒
        return super().fit(train_x, train_y)

    def predict(self, test):
        # 普通に予測
        raw_pred = super().predict(test)
        # yを戻す
        return self._post_process_y(raw_pred)

    def _pre_process_y(self, y):
        """RMSEのlgbmでRMSLEを使った学習を行うために、前処理を行う.

        :yはlog変換する
        """
        return np.log1p(y)

    def _post_process_y(self, pred):
        """RMSEのlgbmでRMSLEを使った学習を行うために、後処理を行う.
        対数をとった分を戻すのと、負の数が出力されることを防ぐために0と比較したmaxをとる.
        """
        return np.maximum(np.exp(pred)-1, 0)

    def return_flag(self):
        return f"{super().return_flag()}_rmsle"
