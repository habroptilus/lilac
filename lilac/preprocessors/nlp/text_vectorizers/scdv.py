from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
import numpy as np
from .text_vectorizer_base import TextVectorizerBase
import json
from lilac.preprocessors.nlp.word_vectorizers.word_vector_factory import WordVectorFactory


class SCDV(TextVectorizerBase):
    def __init__(self,  col_prefix, seed, word_vectorizer, word_vector_size, num_clusters, gmm_max_iter):
        super().__init__(col_prefix, seed)
        self.word_vectorizer_size = word_vector_size

        params = {"vector_size": self.word_vectorizer_size, "seed": seed}
        factory = WordVectorFactory(word_vectorizer, params)

        self.word_vectorizer = factory.run()

        self.num_clusters = num_clusters
        self.percentage = 0.04
        self.gmm_max_iter = gmm_max_iter

    def fit(self, docs):
        """docs: 文字列のリスト"""
        self.word_vectorizer.fit(docs)
        word_vectors = self.word_vectorizer.get_vectors()
        idx, idx_proba = self.cluster_GMM(word_vectors)

        index2word = self.word_vectorizer.get_index2word()

        self.word_centroid_map = dict(zip(index2word, idx))
        self.word_centroid_prob_map = dict(zip(index2word, idx_proba))

        # Computing tf-idf values.
        tfv = TfidfVectorizer(dtype=np.float32)
        tfv.fit(docs)

        featurenames = tfv.get_feature_names()
        idf = tfv._tfidf.idf_

        self.word_idf_dict = {}
        for pair in zip(featurenames, idf):
            self.word_idf_dict[pair[0]] = pair[1]
        return self

    def _transform(self, docs):
        """fitと同じ入力を想定"""
        # gwbowv is a matrix which contains normalised document vectors.
        # Pre-computing probability word-cluster vectors.
        prob_wordvecs = self.get_probability_word_vectors()

        gwbowv = np.zeros((len(docs), self.num_clusters *
                           (self.word_vectorizer_size)), dtype="float32")

        max_no = 0
        min_no = 0

        for i, doc in enumerate(docs):
            words = doc.split(" ")
            gwbowv[i] = self.create_cluster_vector_and_gwbowv(
                prob_wordvecs, words)
            min_no += min(gwbowv[i])
            max_no += max(gwbowv[i])

            if (i+1) % 1000 == 0:
                print(f"Covered : {i+1}")

        min_no = min_no*1.0/len(docs)
        max_no = max_no*1.0/len(docs)
        thres = (abs(max_no) + abs(min_no))/2
        thres = thres*self.percentage

        print("Making sparse...")
        # Set the threshold percentage for making it sparse.

        # Make values of matrices which are less than threshold to zero.
        gwbowv[abs(gwbowv) < thres] = 0

        return gwbowv

    def save(self, path, word_centroid_map_path, word_centroid_prob_map_path, word_idf_dict_path):
        self.word_vectorizer.save(path)
        self.dump_json(word_centroid_map_path, self.word_centroid_map)
        self.dump_json(word_centroid_prob_map_path,
                       self.word_centroid_prob_map)
        self.dump_json(word_idf_dict_path, self.word_idf_dict)

    def load(self, path, word_centroid_map_path, word_centroid_prob_map_path, word_idf_dict_path):
        self.word_vectorizer.load(path)
        self.word_centroid_map = self.load_json(word_centroid_map_path)
        self.word_centroid_prob_map = self.load_json(
            word_centroid_prob_map_path)
        self.word_idf_dict = self.load_json(word_idf_dict_path)

    def dump_json(self, path, d):
        with open(path, 'w') as f:
            json.dump(d, f, indent=4, cls=self.MyEncoder)

    def load_json(self, path):
        with open(path) as f:
            return json.load(f)

    def cluster_GMM(self, word_vectors):
        clf = GaussianMixture(n_components=self.num_clusters,
                              covariance_type="tied", init_params='kmeans', max_iter=self.gmm_max_iter)

        clf.fit(word_vectors)
        idx = clf.predict(word_vectors)
        idx_proba = clf.predict_proba(word_vectors)

        return (idx, idx_proba)

    def get_probability_word_vectors(self):
        prob_wordvecs = {}
        for word in self.word_centroid_map:
            vec = self.word_vectorizer.transform(word)
            if (vec is None) or (word not in self.word_idf_dict):
                continue
            prob_wordvecs[word] = np.zeros(
                self.num_clusters * self.word_vectorizer_size, dtype="float32")

            for index in range(self.num_clusters):
                prob_wordvecs[word][index*self.word_vectorizer_size:(index+1)*self.word_vectorizer_size] = vec * \
                    self.word_centroid_prob_map[word][index]
            prob_wordvecs[word] *= self.word_idf_dict[word]
        return prob_wordvecs

    def create_cluster_vector_and_gwbowv(self, prob_wordvecs, wordlist):
        bag_of_centroids = np.zeros(
            self.num_clusters * self.word_vectorizer_size, dtype="float32")

        for word in wordlist:
            if word in prob_wordvecs:
                bag_of_centroids += prob_wordvecs[word]

        l2_norm = np.linalg.norm(bag_of_centroids, ord=2)
        # norm = np.sqrt(
        #    np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
        if l2_norm != 0:
            bag_of_centroids /= l2_norm

        return bag_of_centroids

    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super().default(obj)
