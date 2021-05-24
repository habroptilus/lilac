from lilac.features.features_base import FeaturesBase
from lilac.preprocessors.nlp.text_vectorizers.text_vectorizer_factory import TextVectorizerFactory


class TextVectorFeatures(FeaturesBase):
    """textをvectorにして特徴量とする.欠損は空文字に変換して渡す."""

    def __init__(self,  vectorizer_flag, target_col, params, features_dir=None):
        self.vectorizer = TextVectorizerFactory(
            vectorizer_flag, target_col, params).run()
        self.target_col = target_col
        super().__init__(features_dir)

    def _fit(self, df):
        self.vectorizer.fit(df[self.target_col].fillna(""))
        return self

    def _transform(self, df):
        return self.vectorizer.transform(df[self.target_col].fillna(""))

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "TextVectorFeatures":
            return self.vectorizer.col_prefix
        else:
            return super().return_flag()
