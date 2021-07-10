import pandas as pd
from lilac.evaluators.evaluator_factory import EvaluatorFactory
from lilac.models.model_factory import ModelFactory
from lilac.trainers.trainer_factory import TrainerFactory
from lilac.validators.cross_validation_runner import CrossValidationRunner
from lilac.validators.folds_generator_factory import FoldsGeneratorFactory
import numpy as np


class MeanFeatureImportanceGetter:
    """fold平均をとったもの"""

    def __init__(self, man_drop_cols, target_col, model_flag, folds_generator_flag,
                 trainer_flag, evaluator_flag, trainer_params, model_params, folds_gen_params):
        self.raw_getter = _LgbmRawFeatureImportancesGetter(man_drop_cols=man_drop_cols,
                                                           target_col=target_col,
                                                           model_flag=model_flag,
                                                           folds_generator_flag=folds_generator_flag,
                                                           trainer_flag=trainer_flag,
                                                           evaluator_flag=evaluator_flag,
                                                           trainer_params=trainer_params,
                                                           model_params=model_params,
                                                           folds_gen_params=folds_gen_params)

    def run(self, train):
        raw = self.raw_getter.run(train)
        raw["importance"] = raw.mean(axis=1)
        return raw[["importance"]]

    def return_flag(self):
        return f"{self.raw_getter.return_flag()}_mean"


class NullImportanceGetter:
    """Null importanceの75%percentileで除してlogをとったものを計算する."""

    def __init__(self, n_trials, man_drop_cols, target_col, model_flag, folds_generator_flag,
                 trainer_flag, evaluator_flag, trainer_params, model_params, folds_gen_params):
        self.getter = MeanFeatureImportanceGetter(man_drop_cols=man_drop_cols,
                                                  target_col=target_col,
                                                  model_flag=model_flag,
                                                  folds_generator_flag=folds_generator_flag,
                                                  trainer_flag=trainer_flag,
                                                  evaluator_flag=evaluator_flag,
                                                  trainer_params=trainer_params,
                                                  model_params=model_params,
                                                  folds_gen_params=folds_gen_params)
        self.n_trials = n_trials
        self.target_col = target_col

    def run(self, train):
        # まず正しい目的変数で実行
        train = train.copy()
        importance = self.getter.run(train)

        for i in range(self.n_trials):
            # 目的変数のシャッフル
            train[self.target_col] = train[self.target_col].sample(
                frac=1).reset_index(drop=True)
            _imp = self.getter.run(train)
            _imp.columns = [f"ni_{i}"]
            importance = pd.concat([importance, _imp], axis=1)
        importance["ni_quantile"] = importance[[
            f"ni_{i}" for i in range(self.n_trials)]].quantile(0.75, axis=1)
        importance["importance"] = np.log1p(
            importance["importance"]/(importance["ni_quantile"]+1))
        return importance[["importance"]]

    def return_flag(self):
        return f"{self.getter.raw_getter.return_flag()}_ni_{self.n_trials}"


class _LgbmRawFeatureImportancesGetter:
    """lgbm系のモデルを使ってfeature importanceを計算する."""

    def __init__(self, man_drop_cols, target_col, model_flag,
                 folds_generator_flag, trainer_flag, evaluator_flag, trainer_params,
                 model_params, folds_gen_params):
        self.runner = LgbmCvRunner(man_drop_cols, target_col, model_flag,
                                   folds_generator_flag, trainer_flag, evaluator_flag, trainer_params,
                                   model_params, folds_gen_params)

    def run(self, df):
        """
        :return selected_cols: selected cols
        """
        output = self.runner.run(df)
        return output["importance"]

    def return_flag(self):
        return self.runner.return_flag()


class LgbmCvRunner:
    """lgbm系のモデルを回す."""

    def __init__(self, man_drop_cols, target_col, model_flag,
                 folds_generator_flag, trainer_flag, evaluator_flag, trainer_params,
                 model_params, folds_gen_params):
        if not model_flag.startswith("lgbm_"):
            raise RuntimeError("Feature selector flag needs to be lgbm model.")
        self.cv = CrossValidationRunner(pred_oof=True, target_col=target_col)
        self.man_drop_cols = man_drop_cols
        self.model_factory = ModelFactory(model_flag, model_params)
        self.folds_generator = FoldsGeneratorFactory().run(
            folds_generator_flag, folds_gen_params)
        self.evaluator = EvaluatorFactory().run(evaluator_flag)
        self.trainer = TrainerFactory(trainer_flag, trainer_params).run()
        self.evaluator_flag = evaluator_flag
        self.target_col = target_col

    def run(self, df):
        """
        :return selected_cols: selected cols
        """
        dropped = df.drop(self.man_drop_cols, axis=1)

        folds = self.folds_generator.run(df)

        output = self.cv.run(dropped, folds, self.model_factory,
                             self.trainer, self.evaluator)
        dfs = [model.get_importance() for model in self.cv.models]
        hoge = pd.concat(dfs, axis=1)
        imp_cols = [f"importance_fold{i}" for i in range(len(dfs))]
        hoge.columns = imp_cols
        output["importance"] = hoge
        return output

    def return_flag(self):
        return f"{self.model_factory.run().return_flag()}_{self.trainer.return_flag()}_{self.folds_generator.return_flag()}_{self.evaluator_flag}_{self.target_col}"
