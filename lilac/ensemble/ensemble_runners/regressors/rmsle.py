from lilac.ensemble.ensemble_runner_base import _EnsembleRunnerBase
import numpy as np
import pandas as pd


class RidgeRmsleEnsemble(_EnsembleRunnerBase):
    """RidgeRmsleを使ってアンサンブルする.元の特徴量は使わず、予測値はlogをとって入力とする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params):
        model_params = {}
        super().__init__(target_col, "rmsle", trainer_flag, "ridge_rmsle",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)

    def _create_datasets(self, output_list, train, test):
        """logをとって入力に入れるのでオーバーライドの必要あり."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            # 予測値の対数をとって入力にする(linearだと効果あるはず)
            train_df[f"pred{i+1}"] = np.log1p(output["oof_raw_pred"])
            test_df[f"pred{i+1}"] = np.log1p(output["raw_pred"])
        train_df[self.target_col] = train[self.target_col]
        return train_df, test_df


class LgbmRmsleEnsemble(_EnsembleRunnerBase):
    """LgbmRmsleを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params):
        model_params = {
            "verbose_eval": 100,
            "num_boost_round": 1000,
            "early_stopping_rounds": 100,
            "lgbm_params": {
                "colsample_bytree": 0.8,
                "max_depth": 5,
                "reg_alpha": 0,
                "reg_lambda": 0,
                "subsample": 0.8,
                "min_child_weight": 1.0,
            },
        }
        super().__init__(target_col, "rmsle", trainer_flag, "lgbm_rmsle",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)


class RandomForestRmsleEnsemble(_EnsembleRunnerBase):
    """RfRmsleを使ってアンサンブルする.元の特徴量は使わない."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params):
        model_params = {"seed": 42}

        super().__init__(target_col, "rmsle", trainer_flag, "rf_rmsle",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)


class LinearRmsleEnsemble(_EnsembleRunnerBase):
    """LinearRmsleを使ってアンサンブルする.元の特徴量は使わず、予測値はlogをとって入力とする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params):
        model_params = {}
        super().__init__(target_col, "rmsle", trainer_flag, "linear_rmsle",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)

    def _create_datasets(self, output_list, train, test):
        """logをとって入力に入れるのでオーバーライドの必要あり."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            # 予測値の対数をとって入力にする(linearだと効果あるはず)
            train_df[f"pred{i+1}"] = np.log1p(output["oof_raw_pred"])
            test_df[f"pred{i+1}"] = np.log1p(output["raw_pred"])
        train_df[self.target_col] = train[self.target_col]
        return train_df, test_df


class AveragingRmsleEnsemble(_EnsembleRunnerBase):
    """AveragingRegressorを使ってアンサンブルする.元の特徴量は使わない"""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params):
        model_params = {}
        super().__init__(target_col, "rmsle", trainer_flag, "avg_regressor",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)

    def _create_datasets(self, output_list, train, test):
        """logをとって入力に入れるのでオーバーライドの必要あり."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            # 予測値の対数をとって入力にする(linearだと効果あるはず)
            train_df[f"pred{i+1}"] = np.log1p(output["oof_raw_pred"])
            test_df[f"pred{i+1}"] = np.log1p(output["raw_pred"])
        train_df[self.target_col] = train[self.target_col]
        return train_df, test_df
