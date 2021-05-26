class FeatureSelector:
    """feature importanceが上位thresholdのものを取り出す."""

    def __init__(self, target_col, threshold=None):
        self.threshold = threshold
        self.target_col = target_col

    def run(self, train, test, importance):
        """importanceが上位threshold以上のものを取り出す.

        thresholdがnoneの場合はimportanceが0でない特徴量を抽出する
        """
        if self.threshold is None:
            print("Selected features with importance > 0 since threshold is None")
            selected_cols = list(
                importance[importance["importance"] > 0].index)
        else:
            selected_cols = list(importance.sort_values(by="importance", ascending=False).head(
                int(len(importance)*self.threshold)).index)

        train = train[selected_cols+[self.target_col]]
        test = test[selected_cols]
        return train, test
