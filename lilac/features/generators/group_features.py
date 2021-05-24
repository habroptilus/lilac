from lilac.features.features_base import FeaturesBase
from xfeat import aggregation
import pandas as pd


class GroupFeaturesBase(FeaturesBase):
    """集約特徴量の基底クラス."""

    def __init__(self,  group_key, target_cols, features_dir=None):
        self.group_key = group_key
        self.target_cols = target_cols

        self.diff_ratio_cols = ["mean", "max", "min",
                                "median"]
        self.agg_only_cols = ["sum", "count",
                              "std", self.MaxMin(), self.Q75Q25()]
        super().__init__(features_dir)

    def _add_diff_ratio(self, data, agg_func):
        result_cols = []
        for group_col in self.target_cols:
            agg_col = f"agg_{agg_func}_{group_col}_grpby_{self.group_key}"
            data[f"{agg_col}_diff"] = data[group_col].values - \
                data[agg_col].values
            data[f"{agg_col}_ratio"] = data[group_col].values / data[agg_col]
            data[f"{agg_col}_diff_ratio"] = data[f"{agg_col}_diff"].values / data[agg_col]
            result_cols.extend(
                [f"{agg_col}_diff", f"{agg_col}_ratio", f"{agg_col}_diff_ratio"])
        return data, result_cols

    class MaxMin:
        def __call__(self, x):
            return x.max()-x.min()

        def __str__(self):
            return "max_min"

    class Q75Q25:
        def __call__(self, x):
            return x.quantile(0.75) - x.quantile(0.25)

        def __str__(self):
            return "q75_q25"


class GroupFeatures(GroupFeaturesBase):
    """集約特徴量を作成する.fitせずtrainとtestを別で計算する."""

    def __init__(self, group_key, target_cols, skip_diff_ratio, features_dir=None):
        self.skip_diff_ratio = skip_diff_ratio
        super().__init__(group_key, target_cols, features_dir)

    def _transform(self, df):
        df, aggregated_cols = aggregation(df,
                                          group_key=self.group_key,
                                          group_values=self.target_cols,
                                          agg_methods=self.diff_ratio_cols+self.agg_only_cols,
                                          )
        if not self.skip_diff_ratio:
            for agg_func in self.diff_ratio_cols:
                df, result_cols = self._add_diff_ratio(df, agg_func)
                aggregated_cols.extend(result_cols)
        return df[aggregated_cols]

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "GroupFeatures":
            return f"{super().return_flag()}_{self.group_key}_"+"_".join(self.target_cols)
        else:
            return super().return_flag()


class GroupFeaturesAppearBoth(GroupFeaturesBase):
    """集約特徴量を作成する.fitしてtransformはfitしたものを使う."""

    def __init__(self, group_key, target_cols, skip_diff_ratio, features_dir=None):
        self.skip_diff_ratio = skip_diff_ratio
        super().__init__(group_key, target_cols, features_dir)

    def _fit(self, df):
        df, aggregated_cols = aggregation(df,
                                          group_key=self.group_key,
                                          group_values=self.target_cols,
                                          agg_methods=self.diff_ratio_cols+self.agg_only_cols,
                                          )

        self._agg = df[~df.duplicated(subset=self.group_key)][[
            self.group_key]+aggregated_cols]

    def _transform(self, df):
        df = pd.merge(df, self._agg, on=self.group_key, how="left")
        aggregated_cols = list(self._agg.columns)
        if not self.skip_diff_ratio:
            for agg_func in self.diff_ratio_cols:
                df, result_cols = self._add_diff_ratio(df, agg_func)
                aggregated_cols.extend(result_cols)
        return df[aggregated_cols].drop(self.group_key, axis=1)

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "GroupFeaturesAppearBoth":
            return f"{super().return_flag()}_{self.group_key}_"+"_".join(self.target_cols)
        else:
            return super().return_flag()


class GroupCorrFeatures(FeaturesBase):
    """グループ分けして相関係数を計算する.fitせずtrainとtestを別で計算する."""

    def __init__(self, group_key, target_cols, features_dir=None):
        """target_colsの指定方法の例
        [["feature_1","feature_2"],["feature_3","feature_4"],["feature_5","feature_6"]]
        """
        self.group_key = group_key
        self.target_cols = target_cols
        super().__init__(features_dir)

    def _transform(self, df):
        dfs = []
        for gv in self.target_cols:
            _df = df.groupby(self.group_key)[gv].corr().unstack().iloc[:, 1].rename(
                f"{self.return_flag()}_{gv[0]}_{gv[1]}")
            dfs.append(pd.DataFrame(_df))
        dfs = pd.concat(dfs, axis=1)
        output_df = pd.merge(df[[self.group_key]], dfs, on=self.group_key, how="left").drop(
            self.group_key, axis=1)
        return output_df

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "GroupCorrFeatures":
            return f"{super().return_flag()}_{self.group_key}_"+"_".join(["_".join(e) for e in self.target_cols])
        else:
            return super().return_flag()


class GroupCountFeatures(FeaturesBase):
    """グループ分けし、あるカラムがある値をとる回数を特徴量に追加する.fitせずtrainとtestを別で計算する."""

    def __init__(self,  group_key, settings, features_dir=None):
        """setttingsの指定方法の例
        [("column_1","value_1"),("column_2","value_2")]
        """
        self.group_key = group_key
        self.settings = settings
        super().__init__(features_dir)

    def _transform(self, df):
        output_df = pd.DataFrame()
        for col, value in self.settings:
            _mapping = df[df[col] == value].groupby(self.group_key).size()
            output_df[f"{self.return_flag()}_{col}_{value}"] = df[self.group_key].map(
                _mapping).fillna(0)
        return output_df

    def return_flag(self):
        if self.__class__.__name__ == "GroupCountFeatures":
            return f"{super().return_flag()}_{self.group_key}_"+"_".join(["_".join(e) for e in self.settings])
        else:
            return super().return_flag()


class GroupUniqueCountFeatures(FeaturesBase):
    """グループ分けして、内部で各カラムごとのunique countsを調べる.fitせずtrainとtest別で計算する."""

    def __init__(self,  group_key, target_cols, features_dir=None):
        self.group_key = group_key
        self.target_cols = target_cols
        super().__init__(features_dir)

    def _transform(self, df):
        for col in self.target_cols:
            _df = df.groupby(self.group_key).nunique()[[col]].rename(
                columns={col: f"ucount_{col}_grpby_{self.group_key}"})
            df = pd.merge(df, _df, on=self.group_key, how="left")
        return df[[f"ucount_{col}_grpby_{self.group_key}" for col in self.target_cols]]

    def return_flag(self):
        if self.__class__.__name__ == "GroupUniqueCountFeatures":
            return f"{super().return_flag()}_{self.group_key}_"+"_".join(self.target_cols)
        else:
            return super().return_flag()


class GroupModeNullFiller(FeaturesBase):
    """グループ分けしてnullのものを内部の最頻値で埋める.trainとtest別."""

    def __init__(self,  group_key, target_cols, features_dir=None):
        self.group_key = group_key
        self.target_cols = target_cols
        super().__init__(features_dir)

    def _transform(self, df):
        for col in self.target_cols:
            mode = df.groupby(self.group_key)[col].apply(lambda x: x.mode())
            merged = pd.merge(df, mode, on=self.group_key, how="left")
            df[f"{self.return_flag()}_{col}"] = merged[f"{col}_x"].fillna(
                merged[f"{col}_y"])
        return df[[f"{self.return_flag()}_{col}" for col in self.target_cols]]

    def return_flag(self):
        if self.__class__.__name__ == "GroupModeNullFiller":
            return f"{super().return_flag()}_{self.group_key}_"+"_".join(self.target_cols)
        else:
            return super().return_flag()
