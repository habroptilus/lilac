from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd


class StandardizingPCA:
    """標準化してからPCAをかける"""

    def __init__(self, n_components, seed, col_prefix="pca"):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.n_components = n_components
        self.col_prefix = col_prefix

    def fit(self, df):
        std_data_df = self.scaler.fit_transform(df)
        self.pca.fit(std_data_df)
        return self

    def transform(self, df):
        data = self.pca.transform(self.scaler.transform(df))
        df = pd.DataFrame(
            data, columns=[f"{self.col_prefix}_{i+1}" for i in range(self.n_components)])
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class KmeansVectorizer:
    """kmeansする."""

    def __init__(self, n_clusters, seed, col_prefix="kmeans"):
        self.n_clusters = n_clusters
        self.seed = seed
        self.col_prefix = col_prefix

    def fit(self, df):
        self.model = KMeans(n_clusters=self.n_clusters,
                            random_state=self.seed).fit(df)
        return self

    def transform(self, df):
        dist = self.model.transform(df)
        probas = dist/dist.sum(axis=1).reshape((-1, 1))
        cols = [f"{self.col_prefix}_{i+1}" for i in range(self.n_clusters)]
        result = pd.DataFrame(probas, columns=cols)
        return result

    def fit_transform(self, df):
        return self.fit(df).transform(df)
