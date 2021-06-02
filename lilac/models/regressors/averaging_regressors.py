from lilac.models.model_base import RegressorBase


class AveragingRegressor(RegressorBase):
    def fit(self, X, y):
        return self

    def _predict(self, df):
        """出力は一次元を想定."""
        return df.mean(axis='columns').to_list()

    def create_flag(self):
        return "avg_reg"
