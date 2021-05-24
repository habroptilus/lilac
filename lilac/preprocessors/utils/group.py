from xfeat import aggregation


class GroupAggregator:
    """集約する.transformしかしない."""

    def __init__(self,  group_key, group_values):
        self.group_key = group_key
        self.group_values = group_values
        self.agg_cols = ["mean", "max", "min",
                         "median", "sum", "count",
                         "std", self.MaxMin(), self.Q75Q25()]

    def transform(self, df):
        df, aggregated_cols = aggregation(df,
                                          group_key=self.group_key,
                                          group_values=self.group_values,
                                          agg_methods=self.agg_cols
                                          )

        return df[[self.group_key] + aggregated_cols].drop_duplicates()

    class MaxMin:
        def __call__(self, x):
            return x.max()-x.min()

        def __str__(self):
            return "max_min"

    class Q75Q25:
        def __call__(self, x):
            return x.quantile(0.75) - x.quantile(0.25)

        def __str__(self):
            return "q75_q25"
