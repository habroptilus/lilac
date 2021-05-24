from lilac.features.generators.combi_features import ArithmeticCombiFeatures
from lilac.features.generators.decomposition_features import KmeansFeatures, PcaFeatures
from lilac.features.generators.group_features import (GroupCorrFeatures, GroupCountFeatures,
                                                      GroupFeatures, GroupFeaturesAppearBoth,
                                                      GroupUniqueCountFeatures)
from lilac.features.generators.lag_features import LagFeatures
from lilac.features.generators.pivot_table_features import PivotTableFeatures
from lilac.features.generators.category_encoding_features import CountEncodingFeatures, TargetEncodingFeatures


class FeaturesFactory:
    def __init__(self, features_dir):
        self.features_dir = features_dir
        self.flag2feature_dict = {
            "group": GroupFeatures,
            "group_appear_both": GroupFeaturesAppearBoth,
            "lag": LagFeatures,
            "kmeans": KmeansFeatures,
            "pca": PcaFeatures,
            "pivot_table": PivotTableFeatures,
            "group_corr": GroupCorrFeatures,
            "group_count": GroupCountFeatures,
            "group_unique_count": GroupUniqueCountFeatures,
            "arith_combi": ArithmeticCombiFeatures,
            "count_enc": CountEncodingFeatures,
            "target_enc": TargetEncodingFeatures
        }

    def run(self, feature_gen_flag, params=None):
        if feature_gen_flag not in self.flag2feature_dict:
            raise Exception(
                f"Invalid feature generator flag. ({feature_gen_flag})")
        if params:
            return self.flag2feature_dict[feature_gen_flag](
                features_dir=self.features_dir, **params)
        else:
            return self.flag2feature_dict[feature_gen_flag](
                features_dir=self.features_dir)

    def register(self, feature_gen_flag, Model):
        if feature_gen_flag in self.flag2feature_dict:
            raise Exception(f"Flag {feature_gen_flag} already exists.")
        self.flag2feature_dict[feature_gen_flag] = Model
