from lilac.models.model_factory import ModelFactory
from lilac.evaluators.evaluator_factory import EvaluatorFactory
from lilac.validators.cross_validation_runner import CrossValidationRunner
from lilac.validators.folds_generator_factory import FoldsGeneratorFactory
from lilac.trainers.trainer_factory import TrainerFactory
import pandas as pd


class _EnsembleRunnerBase:
    """アンサンブルを行う基底クラス.継承して実装する."""

    def __init__(self, target_col, evaluator_flag, trainer_flag, model_flag, folds_generator_flag, trainer_params, model_params, folds_gen_params):
        self.target_col = target_col
        self.cv = CrossValidationRunner(pred_oof=True)
        self.evaluator = EvaluatorFactory(target_col).run(evaluator_flag)
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