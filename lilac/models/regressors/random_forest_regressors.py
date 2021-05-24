
from lilac.models.model_base import RegressorBase
from lilac.models.base.rf_base import _RfRegressor, _RfRmsleRegressor


class RandomForestRmsleRegressor(RegressorBase):
    """RMSLEモデル."""

    def __init__(self, target_col, seed):
        super().__init__(target_col)
        self.seed = seed
        self.model = _RfRmsleRegressor(self.seed)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        self.model.fit(train_x, train_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_rmsle"


class RandomForestRmseRegressor(RegressorBase):
    """RMSEモデル."""

    def __init__(self, target_col, seed):
        super().__init__(target_col)
        self.seed = seed
        self.model = _RfRegressor("mse", self.seed)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)

        self.model.fit(train_x, train_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_rmse"


class RandomForestMaeRegressor(RegressorBase):
    """MAEモデル."""

    def __init__(self, target_col, seed):
        self.seed = seed
        self.model = _RfRegressor("mae", self.seed)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)

        self.model.fit(train_x, train_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_mae"
