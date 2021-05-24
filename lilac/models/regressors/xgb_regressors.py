from lilac.models.model_base import RegressorBase
from lilac.models.base.xgb_base import _XgbRegressor, _XgbRmsleRegressor


class XgbRmseRegressor(RegressorBase):
    """XgboostのRMSEで最適化するモデル.予測時は0以上の値を返す."""

    def __init__(self, verbose_eval, num_boost_round, early_stopping_rounds, xgb_params):
        xgb_params["objective"] = "reg:squarederror"
        xgb_params["eval_metric"] = "rmse"
        self.model = _XgbRegressor(verbose_eval, num_boost_round,
                                   early_stopping_rounds, xgb_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)

        self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return self.model.return_flag()


class XgbRmsleRegressor(RegressorBase):
    """XgboostのRMSLEで最適化するモデル.予測時は0以上の値を返す."""

    def __init__(self, target_col, verbose_eval,  num_boost_round, early_stopping_rounds, xgb_params):
        super().__init__(target_col)
        self.model = _XgbRmsleRegressor(
            verbose_eval, num_boost_round, early_stopping_rounds, xgb_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)

        self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_rmsle"
