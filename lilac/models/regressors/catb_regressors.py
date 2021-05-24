from lilac.models.model_base import RegressorBase
from lilac.models.base.catb_base import _CatbRegressor, _CatbRmsleRegressor


class CatbRmseRegressor(RegressorBase):
    """CatBoost RMSEで最適化"""

    def __init__(self, target_col,  early_stopping_rounds, catb_params):
        super().__init__(target_col)
        self.model = _CatbRegressor(
            "RMSE", early_stopping_rounds, catb_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return self.model.return_flag()


class CatbRmsleRegressor(RegressorBase):
    """CatBoost RMSLEで最適化"""

    def __init__(self, target_col,  early_stopping_rounds, catb_params):
        super().__init__(target_col)
        self.model = _CatbRmsleRegressor(early_stopping_rounds, catb_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_rmsle"
