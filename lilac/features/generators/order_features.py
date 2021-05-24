from lilac.features.features_base import FeaturesBase


class OrderFeatures(FeaturesBase):
    """順番の比率.最後の作品が1で最初の作品が0."""

    def __init__(self, group_key, order_value_col, features_dir=None):
        self.group_key = group_key
        self.order_value_col = order_value_col
        super().__init__(features_dir)

    def _transform(self, df):
        df[self.return_flag()] = df.groupby(self.group_key)[
            self.order_value_col].rank(pct=True)
        return df[[self.return_flag()]]

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "OrderFeatures":
            return f"{super().return_flag()}_{self.group_key}_{self.order_value_col}"
        else:
            return super().return_flag()
