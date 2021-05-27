from sklearn.linear_model import LogisticRegression
from lilac.models.model_base import MultiClassifierBase


class LrMultiClassifier(MultiClassifierBase):

    def __init__(self, target_col, class_weight):
        self.class_weight = class_weight
        self.model = LogisticRegression(
            max_iter=500, class_weight=class_weight)
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
        return f"lr_multi_{self.class_weight}"
