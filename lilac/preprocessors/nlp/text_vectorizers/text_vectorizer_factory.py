from .doc2vectorizer import Doc2Vectorizer
from .scdv import SCDV
from .wordvec_mean_vectorizer import WordvecMeanVectorizer
from .tfidf_vectorizer import TFIDF_Vectorizer
from .lda_vectorizer import LDA_Vectorizer
from .bert_vectorizer import BertVectorizer
from .swem import SWEM


class TextVectorizerFactory:
    def __init__(self, vectorizer_str, col, params):
        self.params = params
        self.required_params = ["col_prefix", "seed"]
        if vectorizer_str == "doc2vec":
            self.required_params.append("vector_size")
            self.Vectorizer = Doc2Vectorizer
            self.params["col_prefix"] = f"{vectorizer_str}_{col}_{params['vector_size']}"
        elif vectorizer_str == "tfidf":
            self.Vectorizer = TFIDF_Vectorizer
            self.params["col_prefix"] = f"{vectorizer_str}_{col}"
        elif vectorizer_str == "lda":
            self.required_params.append("vector_size")
            self.Vectorizer = LDA_Vectorizer
            self.params["col_prefix"] = f"{vectorizer_str}_{col}_{params['vector_size']}"
        elif vectorizer_str == "scdv":
            self.required_params.extend(
                ["word_vector_size", "word_vectorizer", "num_clusters", "gmm_max_iter"])
            self.Vectorizer = SCDV
            self.params["col_prefix"] = f"{vectorizer_str}_{col}_{params['word_vectorizer']}_{params['word_vector_size']}_{params['num_clusters']}_{params['gmm_max_iter']}"
        elif vectorizer_str == "mean":
            self.required_params.extend(
                ["word_vectorizer", "word_vector_size"])
            self.Vectorizer = WordvecMeanVectorizer
            self.params["col_prefix"] = f"{vectorizer_str}_{col}_{params['word_vectorizer']}_{params['word_vector_size']}"
        elif vectorizer_str == "bert":
            self.required_params.extend(
                ["max_len"])
            self.Vectorizer = BertVectorizer
            self.params["col_prefix"] = f"{vectorizer_str}_{col}_{params['max_len']}"
        elif vectorizer_str == "swem":
            self.required_params.extend(
                ["word_vectorizer", "word_vector_size"])
            self.Vectorizer = SWEM
            self.params["col_prefix"] = f"{vectorizer_str}_{col}_{params['word_vectorizer']}_{params['word_vector_size']}"
        else:
            raise Exception("Invalid text vectorizer flag.")

    def run(self):
        params = {e: self.params[e]
                  for e in self.required_params}  # 必要なものだけ取り出す
        return self.Vectorizer(**params)
