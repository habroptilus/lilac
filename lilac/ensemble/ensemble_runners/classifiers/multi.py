from lilac.ensemble.ensemble_runner_base import MultiEnsembleRunnerBase


class LrMultiEnsemble(MultiEnsembleRunnerBase):
    """LrMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params):
        super().__init__(target_col, "f1_macro", trainer_flag, "lr_multi",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)


class AveragingMultiEnsemble(MultiEnsembleRunnerBase):
    """AveragingMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params):
        model_params["group_prefix"] = "pred"
        super().__init__(target_col, "f1_macro", trainer_flag, "avg_multi",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)


class LgbmMultiEnsemble(MultiEnsembleRunnerBase):
    """多クラスlightgbmをつかってアンサンブルする.元の特徴量は使わない.

    多クラス予測確率ベクトルを横に結合して次の層のモデルの特徴に使う.
    """

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params):
        super().__init__(target_col, "f1_macro", trainer_flag, "lgbm_multi",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)
