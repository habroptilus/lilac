class NumericalNullFiller:
    """DataFrameの数値データのカラムの欠損を補完する."""

    def __init__(self, fill_with_mean_cols, substitute_others_list):
        """
        :fill_with_mean_cols データ全体で計算した平均で埋めるカラム
        :substitute_others_list 他の列の値で代用するカラム.
        """

        self.fill_with_mean_cols = fill_with_mean_cols
        self.substitute_others_list = substitute_others_list

    def fit(self, df):
        filled_values = {}
        for col in self.fill_with_mean_cols:
            filled_values[col] = df[col].mean()

        self.filled_values = filled_values
        return self

    def transform(self, df):
        df = df.copy()
        for col, value in self.filled_values.items():
            df[col] = df[col].fillna(value)

        for e in self.substitute_others_list:
            df[e["target"]] = df[e["target"]].fillna(df[e["subs"]])

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
