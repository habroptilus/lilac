import category_encoders as ce


class OnehotEncoder:
    """category encoderを使っているのでnullがあっても平気!"""

    def fit(self, df):
        self.category_cols = df.select_dtypes(include=[object]).columns
        self.encoder = ce.OneHotEncoder(cols=self.category_cols)
        self.encoder.fit(df)
        return self

    def transform(self, df):
        df = df.copy()
        df = self.encoder.transform(df)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class CountEncoder:
    """カテゴリ型のものをすべてcount encodeする"""

    def fit(self, df):
        self.category_cols = df.select_dtypes(include=[object]).columns
        self.encoder = ce.CountEncoder(cols=self.category_cols)
        self.encoder.fit(df)
        return self

    def transform(self, df):
        df = df.copy()
        df = self.encoder.transform(df)
        return df[self.category_cols].add_prefix("ce_")

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class OrdinalEncoder:
    """category encoderを使っているのでnullがあっても平気!"""

    def fit(self, df):
        self.category_cols = df.select_dtypes(include=[object]).columns
        self.encoder = ce.OrdinalEncoder(cols=self.category_cols)
        self.encoder.fit(df)
        return self

    def transform(self, df):
        df = df.copy()
        df = self.encoder.transform(df)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
