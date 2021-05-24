from lilac.models.model_base import RegressorBase
from lilac.models.base.lgbm_base import _LgbmRegressor, _LgbmRmsleRegressor


class LgbmRmsleRegressor(RegressorBase):
    """目的関数がRMSLEのlgbm回帰モデル."""

    def __init__(self, target_col, verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params):
        super().__init__(target_col)
        self.model = _LgbmRmsleRegressor(
            verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)

        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict(test_df)

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()

    def return_flag(self):
        """RMSLEはそのままだとRMSEと同じになってしまうのでつける"""
        return f"{self.model.return_flag()}_rmsle"


class LgbmRmseRegressor(RegressorBase):
    """目的関数がRMSEのlgbm回帰モデル."""

    def __init__(self, target_col, verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params):
        super().__init__(target_col)
        lgbm_params["objective"] = "regression"
        lgbm_params["metrics"] = "rmse"
        self.model = _LgbmRegressor(
            verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict(test_df)

    def return_flag(self):
        return self.model.return_flag()

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()


class LgbmMaeRegressor(RegressorBase):
    """目的関数がMAEのlgbm回帰モデル."""

    def __init__(self, target_col, verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params):
        super().__init__(target_col)
        lgbm_params["objective"] = "regression_l1"
        lgbm_params["metrics"] = "mae"
        self.model = _LgbmRegressor(
            verbose_eval, num_boost_round, early_stopping_rounds, lgbm_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict(test_df)

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()

    def return_flag(self):
        return self.model.return_flag()
