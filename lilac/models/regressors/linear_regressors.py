from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from lilac.models.model_base import RegressorBase


class LinearModel(RegressorBase):
    """普通のlinear regression.欠損やカテゴリ変数に未対応.ensembleには使える."""

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        train_x = self._preprocess_x(train_x)
        train_y = self._preprocess_y(train_y)

        self.model = LinearRegression()
        self.model.fit(train_x, train_y)
        return self

    def _predict(self, x_df):
        pred = np.ravel(self.model.predict(x_df))
        return pred

    def _preprocess_x(self, X):
        return X.values

    def _preprocess_y(self, y):
        return y.reshape((-1, 1))

    def return_flag(self):
        return "linear_rmse"


class LinearPositiveModel(LinearModel):
    """普通のlinear regression.欠損やカテゴリ変数に未対応.正の値を出すように補正だけはする."""

    def _predict(self, x_df):
        pred = super()._predict(x_df)
        return np.maximum(pred, 0)

    def return_flag(self):
        return "linear_rmse_pos"


class LinearRmsle(LinearModel):
    """RMSLEを目的関数にしたlinear regression.欠損やカテゴリ変数に未対応."""

    def _preprocess_y(self, y):
        y = np.log(y+1)
        return super()._preprocess_y(y.reshape((-1, 1)))

    def _predict(self, x_df):
        raw_pred = super()._predict(x_df)
        return np.maximum(np.exp(raw_pred)-1, 0)

    def return_flag(self):
        return "linear_rmsle"


class RidgeModel(RegressorBase):
    """普通のリッジ回帰モデル.欠損やカテゴリ変数に未対応.ensembleには使える."""

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        train_x = self._preprocess_x(train_x)
        train_y = self._preprocess_y(train_y)

        self.model = Ridge(random_state=0)
        self.model.fit(train_x, train_y)
        return self

    def _predict(self, x_df):
        pred = np.ravel(self.model.predict(x_df))
        return pred

    def _preprocess_x(self, X):
        return X.values

    def _preprocess_y(self, y):
        return y.reshape((-1, 1))

    def return_flag(self):
        return "ridge_rmse"


class RidgeRmsle(RidgeModel):
    """RMSLEを目的関数にしたリッジ回帰.欠損やカテゴリ変数に未対応."""

    def _preprocess_y(self, y):
        y = np.log(y+1)
        return super()._preprocess_y(y.reshape((-1, 1)))

    def _predict(self, x_df):
        raw_pred = super()._predict(x_df)
        return np.maximum(np.exp(raw_pred)-1, 0)

    def return_flag(self):
        return "ridge_rmsle"
