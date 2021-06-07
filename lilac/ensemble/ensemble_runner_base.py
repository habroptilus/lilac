from lilac.models.model_factory import ModelFactory
from lilac.evaluators.evaluator_factory import EvaluatorFactory
from lilac.validators.cross_validation_runner import CrossValidationRunner
from lilac.validators.folds_generator_factory import FoldsGeneratorFactory
from lilac.trainers.trainer_factory import TrainerFactory
import pandas as pd
import numpy as np


class _EnsembleRunnerBase:
    """アンサンブルを行う基底クラス.継承して実装する."""

    def __init__(self, target_col, evaluator_flag, trainer_flag, model_flag, folds_generator_flag,
                 trainer_params, model_params, folds_gen_params, use_original_features, drop_cols):
        self.target_col = target_col
        self.cv = CrossValidationRunner(pred_oof=True, target_col=target_col)
        self.evaluator = EvaluatorFactory().run(evaluator_flag)
        self.model_factory = ModelFactory(model_flag, model_params)
        self.trainer = TrainerFactory(trainer_flag, trainer_params).run()
        self.folds_generator = FoldsGeneratorFactory().run(
            folds_generator_flag, folds_gen_params)
        self.use_original_features = use_original_features
        self.drop_cols = drop_cols

    def run(self, output_list, train, test):
        # 前の層の予測からデータセットを作る
        output_based_train, output_based_test = self._create_datasets(
            output_list)

        if self.use_original_features:
            # 元の特徴量を使う場合
            train_dropped = train.drop(self.drop_cols, axis=1)

            train_cols = list(train_dropped.columns)
            print(train_cols)
            train_cols.remove(self.target_col)
            test_dropped = test[train_cols]
            # 元のデータセットと結合する
            train_df = pd.concat([output_based_train, train_dropped], axis=1)
            test_df = pd.concat([output_based_test, test_dropped], axis=1)
        else:
            # 目的変数を設定する
            train_df = output_based_train.copy()
            test_df = output_based_test.copy()
            train_df[self.target_col] = train[self.target_col]

        # oof_predが欠損しているデータはensembleモデルの学習に用いない
        if output_based_train.isnull().any(axis=1).sum():
            print(
                "[Warning] Predictions have None records, so we drop them before ensemble.")
            train_df = train_df.dropna(subset=output_based_train.columns)
            train = train.loc[train.index]

        folds = self.folds_generator.run(train)

        return self.get_result(folds, train_df, test_df)

    def get_result(self, folds, train_df, test_df):
        print(train_df.columns)
        # dropされているかもしれないtrain_df
        dropped_index = train_df.index

        # cvに入力する際にはindexをリセットする必要がある
        result = self.cv.run(train_df.reset_index(drop=True), folds, self.model_factory,
                             self.trainer, self.evaluator)

        # oof_predの生成
        oof_pred_df = pd.DataFrame()
        oof_pred_df["oof_pred"] = [None for _ in range(len(train_df))]
        oof_pred_df.loc[dropped_index, "oof_pred"] = np.array(
            result["oof_pred"])
        result["oof_pred"] = oof_pred_df["oof_pred"].to_list()

        # oof_raw_predの生成
        oof_raw_pred_df = pd.DataFrame()
        oof_raw_pred = np.array(result["oof_raw_pred"])
        if len(oof_raw_pred.shape) == 2:
            cols = [f"oof_pred{i}" for i in range(oof_raw_pred.shape[1])]
            oof_raw_pred_df[cols] = np.full(
                (len(train_df), oof_raw_pred.shape[1]), None)
            oof_raw_pred_df.loc[dropped_index, cols] = oof_raw_pred
            result["oof_raw_pred"] = oof_raw_pred_df[cols].values.tolist()
        elif len(oof_raw_pred.shape) == 1:
            oof_raw_pred_df["oof_pred"] = [None for _ in range(len(train_df))]
            oof_raw_pred_df.loc[dropped_index, "oof_pred"] = oof_raw_pred
            result["oof_raw_pred"] = oof_raw_pred_df["oof_pred"].to_list()
        else:
            raise Exception("Error")

        result["pred"] = list(self.cv.final_output(test_df))
        result["raw_pred"] = list(self.cv.raw_output(test_df))
        return result

    def _create_datasets(self, output_list, train, test):
        raise Exception("Implement please.")


class EnsembleRunnerBase(_EnsembleRunnerBase):
    """回帰、binary, multiクラス分類用アンサンブル基底クラス."""

    def _create_datasets(self, output_list):
        """基本的にはoutput_listを使う.もとの特徴量を使いたい場合はtrain,testから引っ張ってくる."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            oof_raw_pred = np.array(output["oof_raw_pred"])
            raw_pred = np.array(output["raw_pred"])
            if len(oof_raw_pred.shape) == 1:
                raw_pred = np.expand_dims(raw_pred, 1)
                oof_raw_pred = np.expand_dims(oof_raw_pred, -1)

            oof_raw_pred = pd.DataFrame(oof_raw_pred, columns=[
                                        f"pred{i}_{j}" for j in range(oof_raw_pred.shape[1])])

            raw_pred = pd.DataFrame(
                raw_pred, columns=[f"pred{i}_{j}" for j in range(raw_pred.shape[1])])
            train_df = pd.concat([train_df, oof_raw_pred], axis=1)
            test_df = pd.concat([test_df, raw_pred], axis=1)
        return train_df, test_df


class LogSingleEnsembleRunnerBase(EnsembleRunnerBase):
    """入力のlogを取る.RMSLEで線形モデルを使う場合に用いる."""

    def _create_datasets(self, output_list):
        """logをとって入力に入れるのでオーバーライドの必要あり."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            # 予測値の対数をとって入力にする(linearだと効果あるはず)
            train_df[f"pred{i+1}"] = np.log1p(output["oof_raw_pred"])
            test_df[f"pred{i+1}"] = np.log1p(output["raw_pred"])

        return train_df, test_df
