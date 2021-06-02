from lilac.ensemble.ensemble_runner_base import _EnsembleRunnerBase


class RidgeRmseEnsemble(_EnsembleRunnerBase):
    """RidgeRmseを使ってアンサンブルする.元の特徴量は使わない。"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params):
        super().__init__(target_col, "rmse", trainer_flag, "ridge_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)


class LgbmRmseEnsemble(_EnsembleRunnerBase):
    """LgbmRmseを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params):
        super().__init__(target_col, "rmse", trainer_flag, "lgbm_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)


class RandomForestRmseEnsemble(_EnsembleRunnerBase):
    """RfRmseを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params):

        super().__init__(target_col, "rmse", trainer_flag, "rf_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)


class LinearRmseEnsemble(_EnsembleRunnerBase):
    """LinearRmseを使ってアンサンブルする.元の特徴量は使わない"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params):
        super().__init__(target_col, "rmse", trainer_flag, "linear_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)


class AveragingRmseEnsemble(_EnsembleRunnerBase):
    """AveragingRegressorを使ってアンサンブルする.元の特徴量は使わない"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params):
        super().__init__(target_col, "rmse", trainer_flag, "avg_regressor",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)
