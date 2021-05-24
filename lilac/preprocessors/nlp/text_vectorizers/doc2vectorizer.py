from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from .text_vectorizer_base import TextVectorizerBase


class Doc2Vectorizer(TextVectorizerBase):
    def __init__(self, col_prefix, seed, vector_size):
        super().__init__(col_prefix, seed)
        self.vector_size = vector_size

    def fit(self, docs):
        """docs: 文字列のリスト"""
        trainings = [TaggedDocument(words=data.split(), tags=[i])
                     for i, data in enumerate(docs)]
        self.vectorizer = Doc2Vec(
            documents=trainings, vector_size=self.vector_size, seed=self.seed)
        return self

    def _transform(self, docs):
        """fitと同じ入力を想定"""
        doc_words_list = [doc.split(" ") for doc in docs]
        data = []
        for doc_words in doc_words_list:
            data.append(self.vectorizer.infer_vector(doc_words))
        return np.array(data)

    def save(self, path):
        self.vectorizer.save(path)

    def load(self, path):
        self.vectorizer = Doc2Vec.load(path)
