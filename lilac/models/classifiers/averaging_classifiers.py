from lilac.models.model_base import BinaryClassifierBase, MultiClassifierBase
import numpy as np


class AveragingBinaryClassifier(BinaryClassifierBase):
    def fit(self, X, y):
        return self

    def _predict_proba(self, df):
        """出力は一次元でクラス1の予測確率を想定."""
        return df.mean(axis='columns').to_list()

    def create_flag(self):
        return "avg_bin"


class AveragingMultiClassifier(MultiClassifierBase):
    def __init__(self, target_col, group_prefix):
        """
        ex)
        group_prefix=predの場合、
        pred0_*, pred1_*, pred2_*,...,をそれぞれモデルの出力とみて平均する
        """
        self.group_prefix = group_prefix
        super().__init__(target_col)

    def fit(self, X, y):
        return self

    def _predict_proba(self, df):
        """出力はクラス数分の次元でクラスごとの予測確率を想定."""

        num_class = (df.columns.str.startswith(f"{self.group_prefix}0_")).sum()
        if self.target_col in df.columns:
            num_model = int((len(df.columns)-1)/num_class)
        else:
            num_model = int(len(df.columns)/num_class)

        outputs = []
        for i in range(num_model):
            cols = df.columns[df.columns.str.startswith(
                f"{self.group_prefix}{i}")]
            outputs.append(df[cols].values)
        outputs = np.array(outputs)
        output = np.mean(outputs, axis=0)
        return output

    def create_flag(self):
        raise "avg_multi"
