from sklearn.linear_model import LogisticRegression
from lilac.models.model_base import MultiClassifierBase


class LrMultiClassifier(MultiClassifierBase):

    def __init__(self, target_col):
        self.model = LogisticRegression(max_iter=500)
        super().__init__(target_col)

    def fit(self, df, valid):
        y = df[self.target_col]
        X = df.drop([self.target_col], axis=1)
        self.model.fit(X, y)
        return self

    def _predict_proba(self, df):
        """出力はクラス数分の次元でクラスごとの予測確率を想定."""
        return self.model.predict_proba(df)

    def return_flag(self):
        return "lr_multi"
