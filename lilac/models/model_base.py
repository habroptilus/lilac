import numpy as np


class ModelBase:
    """全ての規定モデル."""

    def __init__(self, target_col):
        self.target_col = target_col

    def create_flag(self):
        raise Exception("Implement please.")

    def fit(self, train_df, valid_df):
        raise Exception("Implement please.")

    def get_raw_pred(self):
        raise Exception("Implement please.")

    def predict(self):
        raise Exception("Implement please.")

    def split_df2xy(self, df):
        """xとyに分ける."""
        y = np.array(df[self.target_col])
        X = df.drop(self.target_col, axis=1)
        return X, y


class RegressorBase(ModelBase):
    """回帰モデルのベース."""

    def _predict(self, x_df):
        raise Exception("Implement please.")

    def predict(self, df):
        """target_col以外の特徴量を使って予測する."""
        if self.target_col in df.columns:
            df, _ = self.split_df2xy(df)
        return self._predict(df)

    def get_raw_pred(self, df):
        return self.predict(df)


class BinaryClassifierBase(ModelBase):
    """二値分類モデルのベース."""

    def _predict_proba(self, df):
        """出力は一次元でクラス1の予測確率を想定."""
        raise Exception("Implement please.")

    def predict(self, df):
        """predict_probaを用いて計算.0 or 1を返す."""
        pred_proba = self.predict_proba(df)
        return np.round(pred_proba)

    def predict_proba(self, df):
        if self.target_col in df.columns:
            df, _ = self.split_df2xy(df)
        return self._predict_proba(df)

    def get_raw_pred(self, df):
        return self.predict_proba(df)


class MultiClassifierBase(ModelBase):
    """多値分類モデルのベース."""

    def _predict_proba(self, df):
        """出力はクラス数分の次元でクラスごとの予測確率を想定."""
        raise Exception("Implement please.")

    def predict(self, df):
        """predict_probaを用いて計算.0 ~ class数-1 のどれかを返す."""
        pred_proba = self.predict_proba(df)
        return np.argmax(pred_proba, axis=1)

    def predict_proba(self, df):
        if self.target_col in df.columns:
            df, _ = self.split_df2xy(df)
        return self._predict_proba(df)

    def get_raw_pred(self, df):
        return self.predict_proba(df)
