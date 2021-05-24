from lilac.features.features_base import FeaturesBase
from fasttext import load_model


class TextLengthFeatures(FeaturesBase):
    """指定したカラムの文書長を計算する.スペースで切られている英文前提.NoneはNoneのまま."""

    def __init__(self,  target_cols, features_dir=None):
        self.target_cols = target_cols
        super().__init__(features_dir)

    def _transform(self, df):
        for target_col in self.target_cols:
            df[f"{target_col}_len"] = df[target_col].apply(
                lambda x: len(x.split()) if type(x) != float and x is not None else None)
        return df[[f"{target_col}_len" for target_col in self.target_cols]]

    def return_flag(self):
        if self.__class__.__name__ == "TextLengthFeatures":
            return f"{super().return_flag()}_"+"_".join(self.target_cols)
        else:
            return super().return_flag()


class StringMatchFeatures(FeaturesBase):
    """指定したカラムに指定した文字列が含まれているかを判定する.NoneはNoneのまま."""

    def __init__(self, target_col, target_str, features_dir=None):
        self.target_str = target_str
        self.target_col = target_col
        super().__init__(features_dir)

    def _transform(self, df):
        df[f"{self.target_col}_match_{self.target_str}"] = df[self.target_col].apply(
            lambda x: 1 if type(x) != float and self.target_str in x else 0)
        return df[[f"{self.target_col}_match_{self.target_str}"]]

    def return_flag(self):
        if self.__class__.__name__ == "StringMatchFeatures":
            return f"{super().return_flag()}_{self.target_str}_in_{self.target_col}"
        else:
            return super().return_flag()


class TextLangFeatures(FeaturesBase):
    """指定したカラムの文章の言語を判定して特徴量にする.fasttextをもちいる."""

    def __init__(self,  target_col, features_dir=None):
        self.target_col = target_col
        self.model = load_model("data/lid.176.bin")
        super().__init__(features_dir)

    def _transform(self, df):
        df[f"{self.target_col}_lang"] = df[self.target_col].fillna("").map(
            lambda x: self.model.predict(x.replace("\n", ""))[0][0])
        return df[[f"{self.target_col}_lang"]]
