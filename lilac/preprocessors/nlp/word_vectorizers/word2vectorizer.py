# create word2vec
from gensim.models import Word2Vec


class Word2Vectorizer:
    def __init__(self, vector_size, seed):
        self.vector_size = vector_size
        self.seed = seed

    def fit(self, docs):
        sentences = [doc.split(" ") for doc in docs]
        self.model = Word2Vec(
            sentences,  size=self.vector_size, seed=self.seed)
        self.model.init_sims(replace=True)  # なんだこれ

    def transform(self, word):
        """未知語の場合はNoneを返す.(似ている単語を返すでもいいが...)"""
        if word not in self.model.wv:
            return None
        return self.model.wv[word]

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = Word2Vec.load(path)

    def get_vectors(self):
        return self.model.wv.vectors

    def get_index2word(self):
        return self.model.wv.index2word
