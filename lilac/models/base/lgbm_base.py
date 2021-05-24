import lightgbm as lgb
import pandas as pd
import numpy as np


class _LgbmBase:
    """LGBMのベース"""

    def __init__(self, verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params):
        self.lgbm_params = lgbm_params
        self.verbose_eval = verbose_eval
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, train_x, train_y, valid_x, valid_y):
        train_x = self.object2category(train_x)
        valid_x = self.object2category(valid_x)
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
        self.cols = list(train_x.columns)

        self.model = lgb.train(self.lgbm_params, lgb_train,
                               valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                               verbose_eval=self.verbose_eval,
                               num_boost_round=self.num_boost_round,
                               early_stopping_rounds=self.early_stopping_rounds,
                               )
        return self

    def get_importance(self):
        """特徴量の重要度を出力する."""
        return pd.DataFrame(self.model.feature_importance(importance_type='gain'), index=self.cols, columns=['importance'])

    def object2category(self, df):
        """object型のカラムをcategory型に変換する."""
        object_cols = df.select_dtypes(include=[object]).columns
        for col in object_cols:
            df[col] = df[col].astype("category")
        return df

    def return_flag(self):
        return "lgbm_"+"_".join([str(v) for v in self.lgbm_params.values()])


class _LgbmBinaryClassifier(_LgbmBase):
    """ベースとなるLGBMのbinary分類モデル."""

    def __init__(self, verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params):
        super().__init__(verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params)

    def predict_proba(self, test):
        test = self.object2category(test)
        raw_pred = self.model.predict(
            test, num_iteration=self.model.best_iteration)
        return raw_pred

    def return_flag(self):
        return f"{super().return_flag()}_cls"


class _LgbmClassifier(_LgbmBase):
    """ベースとなるLGBMのclassifierモデル."""

    def __init__(self, verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params):
        super().__init__(verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params)

    def predict_proba(self, test):
        test = self.object2category(test)
        raw_pred = self.model.predict(
            test, num_iteration=self.model.best_iteration)
        return raw_pred

    def return_flag(self):
        return f"{super().return_flag()}_cls"


class _LgbmRegressor(_LgbmBase):
    """ベースとなるLGBMの回帰モデル."""

    def predict(self, test):
        test = self.object2category(test)
        raw_pred = self.model.predict(
            test, num_iteration=self.model.best_iteration)
        return raw_pred

    def return_flag(self):
        return f"{super().return_flag()}_reg"


class _LgbmRmsleRegressor(_LgbmRegressor):
    """LGBMのRMSLEで最適化する回帰モデル."""

    def __init__(self, verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params):
        lgbm_params["objective"] = "regression"
        lgbm_params["metrics"] = "rmse"
        super().__init__(verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params)

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
