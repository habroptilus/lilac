import numpy as np
from .text_vectorizer_base import TextVectorizerBase
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF_Vectorizer(TextVectorizerBase):
    def fit(self, docs):
        """docs: 文字列のリスト"""

        self.tfv = TfidfVectorizer(dtype=np.float32)
        self.tfv.fit(docs)

        return self

    def _transform(self, docs):
        """fitと同じ入力を想定"""
        data = self.tfv.transform(docs).toarray()
        return data
