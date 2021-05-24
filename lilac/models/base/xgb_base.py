from lilac.preprocessors.utils.encoders import OrdinalEncoder
import xgboost as xgb
import numpy as np


class _XgbBase:
    """Xgboostベースクラス."""

    def __init__(self, verbose_eval, num_boost_round, early_stopping_rounds, xgb_params):
        self.xgb_params = xgb_params
        self.encoder = OrdinalEncoder()
        self.num_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval

    def fit(self, train_x, train_y, valid_x, valid_y):
        train_x = self.encoder.fit_transform(train_x)
        valid_x = self.encoder.transform(valid_x)

        train_x = self._pre_process_x(train_x)
        valid_x = self._pre_process_x(valid_x)

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        self.model = xgb.train(self.xgb_params,
                               dtrain,  # 訓練データ
                               self.num_round,  # 設定した学習回数
                               early_stopping_rounds=self.early_stopping_rounds,
                               verbose_eval=self.verbose_eval,
                               evals=watchlist
                               )
        return self

    def _pre_process_x(self, x):
        return x.values

    def return_flag(self):
        return "xgb_"+"_".join([str(v) for v in self.xgb_params.values()])


class _XgbRegressor(_XgbBase):
    """xgboost回帰モデル"""

    def predict(self, test):
        test = self.encoder.transform(test)
        test = self._pre_process_x(test)
        dtest = xgb.DMatrix(test)

        pred = self.model.predict(
            dtest, ntree_limit=self.model.best_ntree_limit)
        return pred.astype(np.float64)

    def return_flag(self):
        return f"{super().return_flag()}_reg"


class _XgbRmsleRegressor(_XgbRegressor):
    """xgboostのRMSLEの内部クラス."""

    def __init__(self, verbose_eval, num_boost_round, early_stopping_rounds, xgb_params):
        xgb_params["objective"] = "reg:squarederror"
        xgb_params["eval_metric"] = "rmse"
        super().__init__(verbose_eval, num_boost_round, early_stopping_rounds, xgb_params)

    def fit(self, train_x, train_y, valid_x, valid_y):
        # yをlog変換
        train_y = self._pre_process_y(train_y)
        valid_y = self._pre_process_y(valid_y)
        # あとはもとのと一緒
        return super().fit(train_x, train_y, valid_x, valid_y)

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
