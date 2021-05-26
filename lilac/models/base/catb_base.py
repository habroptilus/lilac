from catboost import Pool, CatBoostRegressor, CatBoostClassifier
import numpy as np


class _CatbBase:
    """CatBoostの基底クラス."""

    def __init__(self, loss_function, early_stopping_rounds, catb_params):
        self.early_stopping_rounds = early_stopping_rounds
        catb_params["loss_function"] = loss_function
        self.catb_params = catb_params
        self.model = self.get_model()

    def fit(self, train_x, train_y, valid_x, valid_y):
        train_x = self.transform(train_x)
        valid_x = self.transform(valid_x)

        self.categorical_features_indices = np.where(
            train_x.dtypes == object)[0]

        train_pool = Pool(train_x, train_y,
                          cat_features=self.categorical_features_indices)
        validate_pool = Pool(
            valid_x, valid_y, cat_features=self.categorical_features_indices)

        self.model.fit(train_pool,
                       eval_set=validate_pool,
                       early_stopping_rounds=self.early_stopping_rounds,
                       use_best_model=True)
        return self

    def transform(self, df):
        object_cols = df.select_dtypes(include=[object]).columns

        for obj_col in object_cols:
            df[obj_col] = df[obj_col].fillna("NONE_FLAG")

        df = df.fillna(-1e9)
        return df

    def return_flag(self):
        return "catb_"+"_".join([str(v) for v in self.catb_params.values()])

    def get_model(self):
        raise Exception("Implement please.")


class _CatbRegressor(_CatbBase):
    """CatBoostの回帰モデル."""

    def predict(self, test):
        test = self.transform(test)
        test_pool = Pool(test, cat_features=self.categorical_features_indices)
        return self.model.predict(test_pool)

    def return_flag(self):
        return f"{super().return_flag()}_reg"

    def get_model(self):
        return CatBoostRegressor(**self.catb_params)


class _CatbBinaryClassfier(_CatbBase):
    """CatBoostの2クラス分類モデル."""
    Model = CatBoostClassifier

    def __init__(self, early_stopping_rounds, catb_params):
        super().__init__("Logloss", early_stopping_rounds, catb_params)

    def predict(self, test):
        test = self.transform(test)
        test_pool = Pool(test, cat_features=self.categorical_features_indices)
        return self.model.predict(test_pool)

    def predict_proba(self, test):
        test = self.transform(test)
        test_pool = Pool(test, cat_features=self.categorical_features_indices)
        return self.model.predict_proba(test_pool)

    def return_flag(self):
        return f"{super().return_flag()}_bin"

    def get_model(self):
        return CatBoostClassifier(**self.catb_params)


class _CatbMultiClassfier(_CatbBase):
    """CatBoostの多クラス分類モデル."""
    Model = CatBoostClassifier

    def __init__(self, early_stopping_rounds, catb_params):
        super().__init__("MultiClass", early_stopping_rounds, catb_params)

    def predict(self, test):
        test = self.transform(test)
        test_pool = Pool(test, cat_features=self.categorical_features_indices)
        return self.model.predict(test_pool)

    def predict_proba(self, test):
        test = self.transform(test)
        test_pool = Pool(test, cat_features=self.categorical_features_indices)
        return self.model.predict_proba(test_pool)

    def return_flag(self):
        return f"{super().return_flag()}_multi"

    def get_model(self):
        return CatBoostClassifier(**self.catb_params)


class _CatbRmsleRegressor(_CatbRegressor):
    """CatboostのRMSLEで最適化する回帰モデル."""

    def __init__(self,  early_stopping_rounds, catb_params):
        super().__init__("RMSE", early_stopping_rounds, catb_params)

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
