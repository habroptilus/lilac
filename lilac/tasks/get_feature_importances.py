from pathlib import Path

import luigi
import pandas as pd
from luigi.util import requires
from lilac.selectors.importance_getter_factory import ImportanceGetterFactory
from lilac.tasks.feature_generate import FeatureGenerate


@requires(FeatureGenerate)
class GetFeatureImportances(luigi.Task):
    """特徴量選択を行う."""
    # 共通のパラメータ
    target_col = luigi.Parameter()
    folds_generator_flag = luigi.Parameter()
    seed = luigi.IntParameter()
    group_kfolds_col = luigi.Parameter()
    evaluator_flag = luigi.Parameter()
    verbose_eval = luigi.IntParameter()
    n_estimators = luigi.IntParameter()
    early_stopping_rounds = luigi.IntParameter()
    trainer_flag = luigi.Parameter()  # selfは指定不可
    bagging_num = luigi.IntParameter()
    base_class = luigi.IntParameter()
    allow_less_than_base = luigi.BoolParameter()
    class_weight = luigi.Parameter()
    drop_cols = luigi.ListParameter()

    # Feature Selection特有
    importance_flag = luigi.Parameter()
    feature_importance_model_flag = luigi.Parameter()
    fold_num = luigi.IntParameter()
    null_importance_trials = luigi.IntParameter()

    fs_colsample_bytree = luigi.FloatParameter()
    fs_max_depth = luigi.IntParameter()
    fs_reg_alpha = luigi.FloatParameter()
    fs_reg_lambda = luigi.FloatParameter()
    fs_subsample = luigi.FloatParameter()
    fs_min_child_weight = luigi.FloatParameter()

    def output(self):
        d = self.create_dir()
        filenames = ["feature_importances.csv"]
        return [luigi.LocalTarget(d/filename) for filename in filenames]

    def create_dir(self):
        p = Path(self.input()[0].path)
        return p.parent/self.create_flag()

    def create_flag(self):
        params = {
            "man_drop_cols": list(self.drop_cols),
            "target_col": self.target_col,
            "model_flag": self.feature_importance_model_flag,
            "folds_generator_flag": self.folds_generator_flag,
            "trainer_flag": self.trainer_flag,
            "evaluator_flag": self.evaluator_flag,
            "trainer_params": {
                "target_col": self.target_col,
                "bagging_num": self.bagging_num,
                "base_class": self.base_class,
                "seed": self.seed,
                "allow_less_than_base": self.allow_less_than_base
            },
            "n_trials": self.null_importance_trials,
            "model_params": {
                "target_col": self.target_col,
                "verbose_eval": self.verbose_eval,
                "early_stopping_rounds": self.early_stopping_rounds,
                "seed": self.seed,
                "class_weight": self.class_weight,
                "lgbm_params": {
                    "colsample_bytree": self.fs_colsample_bytree,
                    "max_depth": self.fs_max_depth,
                    "reg_alpha": self.fs_reg_alpha,
                    "reg_lambda": self.fs_reg_lambda,
                    "subsample": self.fs_subsample,
                    "min_child_weight": self.fs_min_child_weight,
                    "num_leaves": int(2 ** (self.fs_max_depth) * 0.7),
                    "n_estimators": self.n_estimators
                }
            },
            "folds_gen_params": {
                "fold_num": self.fold_num,
                "seed": self.seed,
                "target_col": self.target_col,
                "key_col": self.group_kfolds_col
            }
        }
        self.imp_getter = ImportanceGetterFactory(
            self.importance_flag).run(params)
        return self.imp_getter.return_flag()

    def run(self):
        print(list(self.drop_cols))
        train = pd.read_csv(self.input()[0].path)

        importances = self.imp_getter.run(train)

        self.output()[0].makedirs()
        importances.to_csv(
            self.output()[0].path, index=True)
