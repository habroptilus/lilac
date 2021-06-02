from lilac.ensemble.ensemble_runner_base import _EnsembleRunnerBase
import pandas as pd
import numpy as np


class LrMultiEnsemble(_EnsembleRunnerBase):
    """LrMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params):
        model_params = {"class_weight": None}
        super().__init__(target_col, "f1_macro", trainer_flag, "lr_multi",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)

    def _create_datasets(self, output_list, train, test):
        """基本的にはoutput_listを使う.もとの特徴量を使いたい場合はtrain,testから引っ張ってくる."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            oof_raw_pred = np.array(output["oof_raw_pred"])
            oof_raw_pred = pd.DataFrame(oof_raw_pred, columns=[
                                        f"pred{i}_{j}" for j in range(oof_raw_pred.shape[1])])
            raw_pred = np.array(output["raw_pred"])
            raw_pred = pd.DataFrame(
                raw_pred, columns=[f"pred{i}_{j}" for j in range(raw_pred.shape[1])])
            train_df = pd.concat([train_df, oof_raw_pred], axis=1)
            test_df = pd.concat([test_df, raw_pred], axis=1)
        train_df[self.target_col] = train[self.target_col]
        return train_df, test_df


class AveragingMultiEnsemble(_EnsembleRunnerBase):
    """AveragingMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params):
        model_params = {"group_prefix": "pred"}
        super().__init__(target_col, "f1_macro", trainer_flag, "avg_multi",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)

    def _create_datasets(self, output_list, train, test):
        """基本的にはoutput_listを使う.もとの特徴量を使いたい場合はtrain,testから引っ張ってくる."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            oof_raw_pred = np.array(output["oof_raw_pred"])
            oof_raw_pred = pd.DataFrame(oof_raw_pred, columns=[
                                        f"pred{i}_{j}" for j in range(oof_raw_pred.shape[1])])
            raw_pred = np.array(output["raw_pred"])
            raw_pred = pd.DataFrame(
                raw_pred, columns=[f"pred{i}_{j}" for j in range(raw_pred.shape[1])])
            train_df = pd.concat([train_df, oof_raw_pred], axis=1)
            test_df = pd.concat([test_df, raw_pred], axis=1)
        train_df[self.target_col] = train[self.target_col]
        return train_df, test_df


class LgbmMultiEnsemble(_EnsembleRunnerBase):
    """多クラスlightgbmをつかってアンサンブルする.元の特徴量は使わない.

    多クラス予測確率ベクトルを横に結合して次の層のモデルの特徴に使う.
    """

    def __init__(self, target_col, folds_generator_flag, folds_gen_params, trainer_flag, trainer_params):
        model_params = {"seed": 42}
        model_params = {
            "verbose_eval": 100,
            "early_stopping_rounds": 20,
            "class_weight": "balanced",
            "lgbm_params": {
                "n_estimators": 2000,
                "colsample_bytree": 0.8,
                "max_depth": 5,
                "reg_alpha": 0,
                "reg_lambda": 0,
                "subsample": 0.8,
                "min_child_weight": 1.0
            },
        }
        super().__init__(target_col, "f1_macro", trainer_flag, "lgbm_multi",
                         folds_generator_flag, trainer_params, model_params, folds_gen_params)

    def _create_datasets(self, output_list, train, test):
        """基本的にはoutput_listを使う.もとの特徴量を使いたい場合はtrain,testから引っ張ってくる."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            oof_raw_pred = np.array(output["oof_raw_pred"])
            oof_raw_pred = pd.DataFrame(oof_raw_pred, columns=[
                                        f"pred{i}_{j}" for j in range(oof_raw_pred.shape[1])])
            raw_pred = np.array(output["raw_pred"])
            raw_pred = pd.DataFrame(
                raw_pred, columns=[f"pred{i}_{j}" for j in range(raw_pred.shape[1])])
            train_df = pd.concat([train_df, oof_raw_pred], axis=1)
            test_df = pd.concat([test_df, raw_pred], axis=1)
        train_df[self.target_col] = train[self.target_col]
        return train_df, test_df
