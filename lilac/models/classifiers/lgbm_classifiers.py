from lilac.models.model_base import BinaryClassifierBase, MultiClassifierBase
from lilac.models.base.lgbm_base import _LgbmClassifier


class LgbmBinaryClassifier(BinaryClassifierBase):
    """目的関数がLoglossのlgbm2値分類モデル.metricは基本loglossだがaccuracyやaucがみたければ指定する."""

    def __init__(self, target_col, verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params):
        super().__init__(target_col)
        lgbm_params["objective"] = "binary"
        lgbm_params["metrics"] = "binary_logloss"
        self.model = _LgbmClassifier(
            verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict_proba(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict_proba(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_bin"

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()


class LgbmMultiClassifier(MultiClassifierBase):
    """目的関数がLoglossのlgbm多値分類モデル.metricは基本loglossだがaccuracyやaucがみたければ指定する."""

    def __init__(self, target_col, verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params, num_class):
        super().__init__(target_col)
        lgbm_params["objective"] = "multiclass"
        lgbm_params["metrics"] = "multi_logloss"
        lgbm_params["num_class"] = num_class
        self.model = _LgbmClassifier(
            verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict_proba(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict_proba(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_multi"

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()
