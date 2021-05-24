from lilac.features.features_base import FeaturesBase
from lilac.preprocessors.utils.decomposition import StandardizingPCA, KmeansVectorizer


class PcaFeatures(FeaturesBase):
    """欠損補完と標準化してからPCAをかける"""

    def __init__(self, target_cols, n_components, seed, features_dir=None):
        self.target_cols = target_cols
        self.n_components = n_components
        self.spca = StandardizingPCA(n_components, seed, self.return_flag())
        super().__init__(features_dir)

    def _fit(self, df):
        df = df[self.target_cols]
        # 重複行を削除
        df = df[~df.duplicated()]
        df = df.fillna(0)
        self.spca.fit(df)
        return self

    def _transform(self, df):
        df = df[self.target_cols]
        df = df.fillna(0)
        return self.spca.transform(df)

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "PcaFeatures":
            return f"{super().return_flag()}_{self.n_components}_"+"_".join(self.target_cols)
        else:
            return super().return_flag()


class KmeansFeatures(FeaturesBase):
    """欠損補完と標準化してからPCAをかける"""

    def __init__(self,  target_cols, n_clusters, seed, features_dir=None):
        self.n_clusters = n_clusters
        self.target_cols = target_cols
        self.kmeans = KmeansVectorizer(n_clusters, seed, self.return_flag())
        super().__init__(features_dir)

    def _fit(self, df):
        df = df[self.target_cols]
        # 重複行を削除
        df = df[~df.duplicated()]
        df = df.fillna(0)
        self.kmeans.fit(df)
        return self

    def _transform(self, df):
        df = df[self.target_cols]
        df = df.fillna(0)
        return self.kmeans.transform(df)

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "KmeansFeatures":
            return f"{super().return_flag()}_{self.n_clusters}_"+"_".join(self.target_cols)
        else:
            return super().return_flag()
