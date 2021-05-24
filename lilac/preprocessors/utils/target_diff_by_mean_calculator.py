import pandas as pd


class TargetDiffByMeanCalculator:
    """target colの、by_colごとにだした平均とそれからの差分を出力"""

    def __init__(self, by_col, target_col):
        self.target = target_col
        self.by_col = by_col

    def fit(self, df):
        dc_mean_by_first = df.groupby(self.by_col).mean()[[self.target]]
        dc_mean_by_first[f"{self.target}_mean_by_{self.by_col}"] = dc_mean_by_first[self.target]
        self.dc_mean_by_first = dc_mean_by_first.drop(self.target, axis=1)
        return self

    def transform(self, df):
        df = df.copy()
        merged = pd.merge(df, self.dc_mean_by_first,
                          on=self.by_col, how="left")
        merged[f"{self.target}_diff_from_{self.by_col}_mean"] = merged[self.target] - \
            merged[f"{self.target}_mean_by_{self.by_col}"]
        return merged[[f"{self.target}_mean_by_{self.by_col}", f"{self.target}_diff_from_{self.by_col}_mean"]]

    def fit_transform(self, df):
        return self.fit(df).transform(df)
