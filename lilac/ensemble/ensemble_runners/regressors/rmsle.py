from lilac.ensemble.ensemble_runner_base import EnsembleRunnerBase, LogSingleEnsembleRunnerBase


class RidgeRmsleEnsemble(LogSingleEnsembleRunnerBase):
    """RidgeRmsleを使ってアンサンブルする.元の特徴量は使わず、予測値はlogをとって入力とする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, use_original_features, drop_cols):
        model_params = {}
        super().__init__(target_col, "rmsle", trainer_flag, "ridge_rmsle",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class LgbmRmsleEnsemble(EnsembleRunnerBase):
    """LgbmRmsleを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmsle", trainer_flag, "lgbm_rmsle",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class RandomForestRmsleEnsemble(EnsembleRunnerBase):
    """RfRmsleを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params,  model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmsle", trainer_flag, "rf_rmsle",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class LinearRmsleEnsemble(LogSingleEnsembleRunnerBase):
    """LinearRmsleを使ってアンサンブルする.元の特徴量は使わず、予測値はlogをとって入力とする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmsle", trainer_flag, "linear_rmsle",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)
