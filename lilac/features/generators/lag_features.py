from lilac.features.features_base import FeaturesBase
import pandas as pd


class LagFeatures(FeaturesBase):
    """差分特徴量を作成する.
    key_colでgroupbyして、行について差分をとるので
    * もとからあるルールで整列されている(例えばYearカラムが昇順で並んでいる)
    * とびとびでない
    などの仮定が置かれている
    """

    def __init__(self,  key_col, target_cols, lag_ranges, features_dir=None,):
        self.key_col = key_col
        self.target_cols = target_cols
        self.lag_ranges = lag_ranges
        super().__init__(features_dir)

    def _transform(self, df):
        output = pd.DataFrame()
        for lag in range(self.lag_ranges[0], self.lag_ranges[1]):
            if lag == 0:
                continue
            diff_df = df.groupby(self.key_col)[self.target_cols].diff(lag)
            diff_df = diff_df.add_prefix(f"Diff_{lag}_")
            output = pd.concat([output, diff_df], axis=1)
            pct_df = df.groupby(self.key_col)[self.target_cols].pct_change(lag)
            pct_df = pct_df.add_prefix(f"Pct_{lag}_")
            output = pd.concat([output, pct_df], axis=1)
        return output

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "LagFeatures":
            return f"{super().return_flag()}_{self.lag_ranges[0]}_{self.lag_ranges[1]}_"+"_".join(self.target_cols)
        else:
            return super().return_flag()
