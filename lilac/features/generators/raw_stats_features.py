"""行方向に統計情報を使って特徴量を作成する."""
from lilac.features.features_base import FeaturesBase
import pandas as pd


class SumInRawFeatures(FeaturesBase):
    """行方向に指定したカラムの和をとる."""

    def __init__(self,  target_cols, features_dir=None):
        self.target_cols = target_cols
        super().__init__(features_dir)

    def _transform(self, df):
        df = df[self.target_cols]
        return pd.DataFrame(df.sum(axis=1), columns=[f"{self.return_flag()}"])

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "SumInRawFeatures":
            return f"{super().return_flag()}_" + "_".join(self.target_cols)
        else:
            return super().return_flag()
