from lilac.models.model_factory import ModelFactory
from lilac.evaluators.evaluator_factory import EvaluatorFactory
from lilac.validators.cross_validation_runner import CrossValidationRunner
from lilac.validators.folds_generator_factory import FoldsGeneratorFactory
from lilac.trainers.trainer_factory import TrainerFactory
import pandas as pd
import numpy as np


class _EnsembleRunnerBase:
    """アンサンブルを行う基底クラス.継承して実装する."""

    def __init__(self, target_col, evaluator_flag, trainer_flag, model_flag, folds_generator_flag, trainer_params, model_params, folds_gen_params):
        self.target_col = target_col
        self.cv = CrossValidationRunner(pred_oof=True)
        self.evaluator = EvaluatorFactory(target_col).run(evaluator_flag)
        model_params["target_col"] = target_col
        self.model_factory = ModelFactory(model_flag, model_params)
        self.trainer = TrainerFactory(trainer_flag, trainer_params).run()
        self.folds_generator = FoldsGeneratorFactory().run(
            folds_generator_flag, folds_gen_params)

    def run(self, output_list, train, test):
        train_df, test_df = self._create_datasets(output_list, train, test)
        folds = self.folds_generator.run(train)
        result = self.cv.run(train_df, folds, self.model_factory,
                             self.trainer, self.evaluator)
        result["pred"] = list(self.cv.final_output(test_df))
        result["raw_pred"] = list(self.cv.raw_output(test_df))
        return result

    def _create_datasets(self, output_list, train, test):
        """基本的にはoutput_listを使う.もとの特徴量を使いたい場合はtrain,testから引っ張ってくる."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            train_df[f"pred{i+1}"] = output["oof_raw_pred"]
            test_df[f"pred{i+1}"] = output["raw_pred"]
        train_df[self.target_col] = train[self.target_col]
        return train_df, test_df


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
