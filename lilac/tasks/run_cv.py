import json
from pathlib import Path

import luigi
import pandas as pd
from luigi.util import inherits
from lilac.evaluators.evaluator_factory import EvaluatorFactory
from lilac.models.model_factory import ModelFactory
from lilac.selectors.feature_selector import FeatureSelector
from lilac.tasks.feature_generate import FeatureGenerate
from lilac.tasks.get_feature_importances import GetFeatureImportances
from lilac.trainers.trainer_factory import TrainerFactory
from lilac.utils.utils import MyEncoder
from lilac.validators.cross_validation_runner import CrossValidationRunner
from lilac.validators.folds_generator_factory import FoldsGeneratorFactory


@inherits(FeatureGenerate)
@inherits(GetFeatureImportances)
class RunCv(luigi.Task):
    """CVを実行してtrain, evaluate, predictを行う."""

    threshold = luigi.FloatParameter(default=None)
    model_flag = luigi.Parameter()
    colsample_bytree = luigi.FloatParameter(default=0.8)
    max_depth = luigi.IntParameter(default=5)
    reg_alpha = luigi.FloatParameter(default=0)
    reg_lambda = luigi.FloatParameter(default=0)
    subsample = luigi.FloatParameter(default=0.8)
    min_child_weight = luigi.FloatParameter(default=1.0)
    lr = luigi.FloatParameter(default=0.1)
    random_strength = luigi.IntParameter(default=1)
    bagging_temperature = luigi.FloatParameter(default=0.1)
    od_type = luigi.Parameter(default="IncToDec")
    od_wait = luigi.IntParameter(default=10)

    def requires(self):
        return [self.clone(FeatureGenerate), self.clone(GetFeatureImportances)]

    def output(self):
        d = self.create_dir()
        return luigi.LocalTarget(d/"result.json")

    def create_dir(self):
        p = Path(self.input()[1][0].path)
        return p.parent/self.create_flag()

    def create_flag(self):
        model_params = {
            "target_col": self.target_col,
            "verbose_eval": self.verbose_eval,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "num_class": self.num_class,
            "lgbm_params": {
                "colsample_bytree": self.colsample_bytree,
                "max_depth": self.max_depth,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "subsample": self.subsample,
                "min_child_weight": self.min_child_weight,
                "num_leaves": int(2 ** (self.max_depth) * 0.7)
            },
            "xgb_params": {
                'random_state': self.seed,
                "colsample_bytree": self.colsample_bytree,
                "max_depth": self.max_depth,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "subsample": self.subsample,
                "min_child_weight": self.min_child_weight
            },
            "catb_params": {
                'depth': self.max_depth,
                'learning_rate': self.lr,
                'random_strength': self.random_strength,
                'bagging_temperature': self.bagging_temperature,
                'od_type': self.od_type,
                'od_wait': self.od_wait,
                "random_seed": self.seed
            }
        }
        trainer_params = {
            "target_col": self.target_col,
            "bagging_num": self.bagging_num,
            "base_class": self.base_class,
            "seed": self.seed,
            "allow_less_than_base": self.allow_less_than_base
        }
        folds_gen_params = {"fold_num": self.fold_num, "seed": self.seed,
                            "target_col": self.target_col, "key_col": self.group_kfolds_col}

        self.model_factory = ModelFactory(self.model_flag, model_params)
        self.trainer_factory = TrainerFactory(
            self.trainer_flag, trainer_params)
        self.folds_generator = FoldsGeneratorFactory().run(
            self.folds_generator_flag, folds_gen_params)

        return f"{self.model_factory.run().return_flag()}_{self.trainer_factory.run().return_flag()}_"
        +f"{self.folds_generator.return_flag()}_{self.evaluator_flag}_{self.threshold}"

    def run(self):
        before_dropped = pd.read_csv(self.input()[0][0].path)
        test = pd.read_csv(self.input()[0][1].path)
        importance = pd.read_csv(self.input()[1][0].path, index_col=0)

        # 特徴量削減
        selector = FeatureSelector(self.target_col, self.threshold)
        train, test = selector.run(before_dropped, test, importance)
        print(
            f"Selected columns : {len(before_dropped.columns)-1} -> {len(train.columns)-1}")

        # 特徴量削減前のデータでfoldsを計算
        # feature selectと同じコード
        folds = self.folds_generator.run(before_dropped)

        evaluator_factory = EvaluatorFactory(self.target_col)
        evaluator = evaluator_factory.run(self.evaluator_flag)
        trainer = self.trainer_factory.run()
        cv = CrossValidationRunner(pred_oof=True)
        output = cv.run(train, folds, self.model_factory, trainer, evaluator)
        output["raw_pred"] = list(cv.raw_output(test))
        output["pred"] = list(cv.final_output(test))

        with self.output().open("w") as f:
            json.dump(output, f, cls=MyEncoder)
