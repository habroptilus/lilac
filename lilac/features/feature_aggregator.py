import pandas as pd


class FeaturesAggregator:
    def __init__(self, settings, factory):
        self.feature_gen_list = [factory.run(
            **setting) for setting in settings]

    def run(self, train_data, test_data):
        train_df = train_data.copy()
        test_df = test_data.copy()
        features_n = len(self.feature_gen_list)
        for i, feature_gen in enumerate(self.feature_gen_list):
            print(f"[{i+1}/{features_n}] {feature_gen.return_flag()}")
            train, test = feature_gen.run(train_data, test_data)
            if (len(set(train_df.columns) & set(train.columns)) > 0) or (len(set(test_df.columns) & set(test.columns)) > 0):
                raise Exception(
                    "Aggregation error. Duplicated cols are about to be added.")
            train_df = pd.concat([train_df, train], axis=1)
            test_df = pd.concat([test_df, test], axis=1)

        return train_df, test_df
