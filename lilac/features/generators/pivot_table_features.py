from lilac.features.features_base import FeaturesBase
from lilac.preprocessors.utils.pivot_table import PivotTable
import pandas as pd


class PivotTableFeatures(FeaturesBase):
    """trainとtestで別々に処理するのでfitなし.

    key_colの値に対して、ベクトルを作成して元のdfとjoinする.
    ベクトルは(emb_col_nameの水準数*values_colsの長さ)の次元が得られる
    [CAUTION] : trainとtestでemb_col_nameの水準数が異なった場合に違うtrainとtestで次元数の異なる出力になってしまう可能性あり.
    """

    def __init__(self,  key_col, emb_col_name, values_cols, features_dir=None):
        self.pt = PivotTable(key_col, emb_col_name, values_cols)
        super().__init__(features_dir)

    def _transform(self, df):
        all_df = self.pt.transform(df)
        output_df = pd.merge(df[[self.key_col]], all_df,
                             on=self.key_col, how="left")
        return output_df.drop(self.key_col, axis=1)

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "PivotTableFeatures":
            return f"{super().return_flag()}_{self.pt.return_flag()}"
        else:
            return super().return_flag()
