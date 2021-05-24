from lilac.features.features_base import FeaturesBase, TrainOnlyFeatureBase
from lilac.preprocessors.utils.encoders import CountEncoder
from xfeat import TargetEncoder
from lilac.validators.folds_generator_factory import FoldsGeneratorFactory
from sklearn.model_selection import KFold
import pandas as pd


class CountEncodingFeatures(FeaturesBase):
    """count encodingする特徴量."""

    def __init__(self, target_cols, features_dir=None):
        self.target_cols = target_cols
        self.encoder = CountEncoder()
        super().__init__(features_dir)

    def _fit(self, df):
        self.encoder.fit(df[self.target_cols])
        return self

    def _transform(self, df):
        return self.encoder.transform(df[self.target_cols])

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "CountEncodingFeatures":
            return f"{super().return_flag()}_" + "_".join(self.target_cols)
        else:
            return super().return_flag()


class TargetEncodingFeatures(FeaturesBase):
    """target encodingする特徴量.kfoldしかできない"""

    def __init__(self, input_cols, target_col, features_dir=None):
        self.input_cols = input_cols
        self.target_col = target_col
        super().__init__(features_dir)

    def fit_transform(self, df):
        df = df.copy()
        fold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.encoder = TargetEncoder(input_cols=self.input_cols,
                                     target_col=self.target_col,
                                     fold=fold,
                                     output_suffix="_te")
        result = self.encoder.fit_transform(df)
        return result[[f"{col}_te_{self.target_col}" for col in self.input_cols]]

    def _transform(self, df):
        return self.encoder.transform(df)[[f"{col}_te" for col in self.input_cols]]

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "TargetEncodingFeatures":
            return f"{super().return_flag()}_{self.target_col}_" + "_".join(self.input_cols)
        else:
            return super().return_flag()


class GetDummiedCategoriesTrainOnly(TrainOnlyFeatureBase):
    """trainのみに現れる特徴量について、カテゴリを展開する.(onehot encoding的な)"""

    def __init__(self, target_cols, features_dir=None):
        self.target_cols = target_cols
        super().__init__(features_dir)

    def _fit_transform(self, df):
        return pd.get_dummies(df[self.target_cols], drop_first=True, dummy_na=True)


class _TargetEncodingFeatures(FeaturesBase):
    """target encodingする特徴量
    :これだとfoldの部分が既にsplitされているのでだめ。。一応残す
    """

    def __init__(self, input_cols, target_col, fold_params, fold_flag, features_dir=None):
        self.fold_gen = FoldsGeneratorFactory().run(fold_flag, fold_params)
        self.input_cols = input_cols
        self.target_col = target_col
        super().__init__(features_dir)

    def _fit(self, df):
        fold = self.fold_gen.run(df)
        self.encoder = TargetEncoder(input_cols=self.input_cols,
                                     target_col=self.target_col, fold=fold)

        self.encoder.fit(df[self.input_cols])
        return self

    def _transform(self, df):
        return self.encoder.transform(df[self.input_cols])

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "TargetEncodingFeatures":
            return f"{super().return_flag()}_{self.fold_gen.return_flag()}_{self.target_col}_" + "_".join(self.input_cols)
        else:
            return super().return_flag()
