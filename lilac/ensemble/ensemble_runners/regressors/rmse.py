from lilac.ensemble.ensemble_runner_base import EnsembleRunnerBase


class RidgeRmseEnsemble(EnsembleRunnerBase):
    """RidgeRmseを使ってアンサンブルする.元の特徴量は使わない。"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmse", trainer_flag, "ridge_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class LgbmRmseEnsemble(EnsembleRunnerBase):
    """LgbmRmseを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmse", trainer_flag, "lgbm_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class RandomForestRmseEnsemble(EnsembleRunnerBase):
    """RfRmseを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):

        super().__init__(target_col, "rmse", trainer_flag, "rf_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class LinearRmseEnsemble(EnsembleRunnerBase):
    """LinearRmseを使ってアンサンブルする.元の特徴量は使わない"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmse", trainer_flag, "linear_rmse",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)


class AveragingRmseEnsemble(EnsembleRunnerBase):
    """AveragingRegressorを使ってアンサンブルする.元の特徴量は使わない"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params, model_params, use_original_features, drop_cols):
        super().__init__(target_col, "rmse", trainer_flag, "avg_regressor",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params, use_original_features, drop_cols)
