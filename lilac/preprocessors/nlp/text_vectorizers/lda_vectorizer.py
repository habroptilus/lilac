from .text_vectorizer_base import TextVectorizerBase
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation


class LDA_Vectorizer(TextVectorizerBase):
    def __init__(self, col_prefix, seed, vector_size):
        super().__init__(col_prefix, seed)
        self.vector_size = vector_size

    def transform(self, docs):
        """ldaだけはどのカテゴリに属するかが大事だと思うので、そのカラムも入れる"""
        lda_data = super().transform(docs)
        result = lda_data.copy()
        result[f"{self.col_prefix}_idmax"] = lda_data.idxmax(axis=1)
        result[f"{self.col_prefix}_idmin"] = lda_data.idxmin(axis=1)
        return result

    def fit(self, docs):
        """docs: 文字列のリスト"""
        self.tfidf_vec = TfidfVectorizer().fit(docs)
        X_train = self.tfidf_vec.transform(docs)

        self.lda = LatentDirichletAllocation(
            n_components=self.vector_size,  random_state=self.seed)
        self.lda.fit(X_train)
        return self

    def _transform(self, docs):
        """fitと同じ入力を想定"""
        data = self.tfidf_vec.transform(docs)
        return self.lda.transform(data)
