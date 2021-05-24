from .word2vectorizer import Word2Vectorizer
from .fasttext_vectorizer import FasttextVectorizer


class WordVectorFactory:
    def __init__(self, vectorizer_str, params):
        self.params = params
        if vectorizer_str == "w2v":
            self.Vectorizer = Word2Vectorizer
        elif vectorizer_str == "fasttext":
            self.Vectorizer = FasttextVectorizer
        else:
            raise Exception("Invalid word vectorizer flag.")

    def run(self):
        return self.Vectorizer(**self.params)
