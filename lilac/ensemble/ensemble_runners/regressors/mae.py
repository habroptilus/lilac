from lilac.ensemble.ensemble_runner_base import EnsembleRunnerBase


class LgbmMaeEnsemble(EnsembleRunnerBase):
    """LgbmMaeを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "mae", trainer_flag, "lgbm_mae",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)
