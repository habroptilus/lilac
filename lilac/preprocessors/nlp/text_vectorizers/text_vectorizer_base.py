import pandas as pd


class TextVectorizerBase:
    """text vectorizerの基底クラス."""

    def __init__(self, col_prefix, seed):
        self.col_prefix = col_prefix
        self.seed = seed

    def transform(self, docs):
        data = self._transform(docs)
        return pd.DataFrame(data, columns=[self.col_prefix+f"_{i+1}" for i in range(data.shape[1])])

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def fit(self, docs):
        raise Exception("Not implemented.")

    def _transform(self, docs):
        """継承したらこれを実装してくれ.np.arrayでかえすように."""
        raise Exception("Not implemented.")

    def save(self, path):
        raise Exception("Not implemented.")

    def load(self, path):
        raise Exception("Not implemented.")
