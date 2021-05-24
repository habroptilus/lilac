import numpy as np
from lilac.preprocessors.nlp.text_vectorizers.text_vectorizer_base import TextVectorizerBase
from lilac.preprocessors.nlp.word_vectorizers.word_vector_factory import WordVectorFactory


class WordvecMeanVectorizer(TextVectorizerBase):
    def __init__(self, col_prefix, seed, word_vector_size,  word_vectorizer):
        super().__init__(col_prefix, seed)
        params = {"vector_size": word_vector_size, "seed": seed}
        factory = WordVectorFactory(word_vectorizer, params)

        self.word_vectorizer = factory.run()
        self.vector_size = word_vector_size

    def fit(self, docs):
        """docs: 文字列のリスト"""
        self.word_vectorizer.fit(docs)
        return self

    def _transform(self, docs):
        """fitと同じ入力を想定"""
        data = []
        for doc in docs:
            words = doc.split(" ")
            doc_data = []
            for word in words:
                vector = self.word_vectorizer.transform(word)
                if vector is not None:
                    doc_data.append(vector)
            if len(doc_data) == 0:
                # 一単語もベクトルにできなかった場合は0ベクトルを返す.
                mean_vec = np.zeros(self.vector_size)
            else:
                mean_vec = np.array(doc_data).mean(axis=0)
            data.append(mean_vec)
        return np.array(data)

    def save(self, path):
        self.word_vectorizer.save(path)

    def load(self, path):
        self.word_vectorizer.load(path)
