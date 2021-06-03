import json
from pathlib import Path

import luigi
import pandas as pd
from lilac.features.feature_aggregator import FeaturesAggregator
from lilac.features.features_factory import FeaturesFactory
from features import additional_features


class FeatureGenerate(luigi.Task):
    """特徴量を作成、結合し、学習に用いる形まで変換する.
    教師あり学習なら書き換える必要なし.
    半教師あり学習ならラベルなしデータを除去する処理を書く必要がある.
    """

    result_dir = luigi.Parameter()
    luigi_dir = luigi.Parameter()
    features_dir = luigi.Parameter()
    features_conf_key = luigi.Parameter()
    settings_path = luigi.Parameter()

    def output(self):
        d = self.create_dir()
        filenames = ["train.csv", "test.csv"]
        return [luigi.LocalTarget(d/filename) for filename in filenames]

    def create_dir(self):
        return Path(f"{self.result_dir}/{self.luigi_dir}/{self.features_conf_key}")

    def run(self):
        with Path(self.settings_path).open() as f:
            features_conf = json.load(f)["features"]

        df_train = pd.read_csv(f"{self.result_dir}/{self.luigi_dir}/train.csv")
        df_test = pd.read_csv(f"{self.result_dir}/{self.luigi_dir}/test.csv")

        features_factory = FeaturesFactory(
            Path(self.result_dir)/self.features_dir)

        for flag, model in additional_features.items():
            features_factory.register(flag, model)

        agg = FeaturesAggregator(
            features_conf[self.features_conf_key], features_factory)

        df_train, df_test = agg.run(df_train, df_test)

        self.output()[0].makedirs()
        df_train.to_csv(self.output()[0].path, index=False)
        df_test.to_csv(self.output()[1].path, index=False)
