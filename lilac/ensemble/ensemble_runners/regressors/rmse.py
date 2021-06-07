from lilac.ensemble.ensemble_runner_base import SingleEnsembleRunnerBase


class RidgeRmseEnsemble(SingleEnsembleRunnerBase):
    """RidgeRmseを使ってアンサンブルする.元の特徴量は使わない。"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmse", trainer_flag, "ridge_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class LgbmRmseEnsemble(SingleEnsembleRunnerBase):
    """LgbmRmseを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmse", trainer_flag, "lgbm_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class RandomForestRmseEnsemble(SingleEnsembleRunnerBase):
    """RfRmseを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):

        super().__init__(target_col, "rmse", trainer_flag, "rf_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class LinearRmseEnsemble(SingleEnsembleRunnerBase):
    """LinearRmseを使ってアンサンブルする.元の特徴量は使わない"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmse", trainer_flag, "linear_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class AveragingRmseEnsemble(SingleEnsembleRunnerBase):
    """AveragingRegressorを使ってアンサンブルする.元の特徴量は使わない"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmse", trainer_flag, "avg_regressor",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)
