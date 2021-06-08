from lilac.ensemble.ensemble_runner_base import EnsembleRunnerBase


class LrMultiEnsemble(EnsembleRunnerBase):
    """LrMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "f1_macro", trainer_flag, "lr_multi",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class AveragingMultiEnsemble(EnsembleRunnerBase):
    """AveragingMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        model_params["group_prefix"] = "pred"
        super().__init__(target_col, "f1_macro", trainer_flag, "avg_multi",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class LgbmMultiEnsemble(EnsembleRunnerBase):
    """多クラスlightgbmをつかってアンサンブルする.元の特徴量は使わない.

    多クラス予測確率ベクトルを横に結合して次の層のモデルの特徴に使う.
    """

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "f1_macro", trainer_flag, "lgbm_multi",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)
