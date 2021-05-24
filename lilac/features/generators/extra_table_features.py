from lilac.features.features_base import FeaturesBase
import pandas as pd
import numpy as np


class _ExtraTableJoin(FeaturesBase):
    """外部テーブルを結合する."""

    def __init__(self, csv_path, join_on, features_dir=None):
        self.csv_path = csv_path
        self.join_on = join_on
        super().__init__(features_dir)

    def _transform(self, df):
        self.ex_table = self._preprocess(pd.read_csv(self.csv_path))
        return self._join(df)

    def _join(self, df):
        """dfとself.ex_tableを使ってjoinし、カラムを絞って返すように実装する."""
        raise Exception("継承してつかってね")

    def _preprocess(self, ex_table):
        """外部テーブル自体に前処理をする場合はこちらもオーバーライドする."""
        return ex_table


class ExtraTableJoin(_ExtraTableJoin):
    """普通のテーブルを結合する場合."""

    def _join(self, df):
        df = pd.merge(df, self.ex_table, on=self.join_on, how="left")
        return_cols = list(self.ex_table.columns)
        if self.join_on in return_cols:
            return_cols.remove(self.join_on)
        return df[return_cols]


class ExtraTableMultiJoin(_ExtraTableJoin):
    """複数対応テーブルをmultihotで結合する場合はこちらを継承する."""

    def __init__(self,  csv_path, join_on, target_cols, weight_col=None, features_dir=None):
        self.target_cols = target_cols
        self.weight_col = weight_col
        super().__init__(features_dir=features_dir, csv_path=csv_path, join_on=join_on)

    def _join(self, df):
        """クロス表をつかってmultihotにして返す."""
        cols = []
        for target_col in self.target_cols:
            if self.weight_col:
                cross_tab = pd.crosstab(
                    self.ex_table[self.join_on], self.ex_table[target_col], values=self.ex_table[self.weight_col], aggfunc=np.sum)

                cross_tab = cross_tab.fillna(0)
            else:
                cross_tab = pd.crosstab(
                    self.ex_table[self.join_on], self.ex_table[target_col])
            cross_tab = cross_tab.add_prefix(
                f"{self.return_flag()}_{target_col}")
            df = pd.merge(df, cross_tab, on=self.join_on, how="left")
            cols.extend(list(cross_tab.columns))
        return df[cross_tab.columns]

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "ExtraTableMultiJoin":
            return f"{super().return_flag()}_{self.weight_col}_" + "_".join(self.target_cols)
        else:
            return super().return_flag()
