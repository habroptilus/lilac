from lilac.features.features_base import FeaturesBase
import pandas as pd


class DateTimeFeatures(FeaturesBase):
    """pd.to_datetimeで変換できる形のstringを受け取り、timestampと年月日を返す.(欠損があっても大丈夫.)"""

    def __init__(self, target_col, features_dir=None):
        self.target_col = target_col
        super().__init__(features_dir)

    def _transform(self, df):
        # {self.target_col}_dateはdatetimeに変換できたのでこれを利用
        df[f"{self.target_col}_datetime"] = pd.to_datetime(df[self.target_col])

        # datetimeからtimestampに
        df[f"{self.target_col}_ts"] = df[f"{self.target_col}_datetime"].apply(
            lambda x: x.timestamp()/1e9 if type(x) != type(pd.NaT) else None)

        # datetimeから年月日を計算
        df[f"{self.target_col}_year"] = df[f"{self.target_col}_datetime"].apply(
            lambda x: x.year if type(x) != type(pd.NaT) else None)
        df[f"{self.target_col}_month"] = df[f"{self.target_col}_datetime"].apply(
            lambda x: x.month if type(x) != type(pd.NaT) else None).astype(str)
        df[f"{self.target_col}_day"] = df[f"{self.target_col}_datetime"].apply(
            lambda x: x.day if type(x) != type(pd.NaT) else None).astype(str)
        return df[[f"{self.target_col}_{e}" for e in ["year", "month", "day", "ts"]]]

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "DateTimeFeatures":
            return f"{super().return_flag()}_{self.target_col}"
        else:
            return super().return_flag()
