from lilac.selectors.lgbm_importance_getters import LgbmCvRunner
import pandas as pd


class LgbmAdversarialValidator:
    """Lgbmを用いてtrainとtestの分布の差を調べる."""
    target_col = "is_test"
    model_flag = "lgbm_bin"
    evaluator_flag = "auc"

    def __init__(self, trainer_flag,
                 folds_generator_flag, man_drop_cols, original_target_col,
                 group_kfolds_col="gameID", verbose_eval=100, num_boost_round=2000,
                 early_stopping_rounds=20, num_class=8, colsample_bytree=0.8, max_depth=5,
                 reg_alpha=0, reg_lambda=0, subsample=0.8, min_child_weight=1, bagging_num=10,
                 base_class=3, seed=42, allow_less_than_base=True, fold_num=5):
        model_params = {
            "target_col": self.target_col,
            "verbose_eval": verbose_eval,
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
            "lgbm_params": {
                "colsample_bytree": colsample_bytree,
                "max_depth": max_depth,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "subsample": subsample,
                "min_child_weight": min_child_weight,
                "num_leaves": int(2 ** (max_depth) * 0.7)
            }
        }
        trainer_params = {
            "target_col": self.target_col,
            "bagging_num": bagging_num,
            "base_class": base_class,
            "seed": seed,
            "allow_less_than_base": allow_less_than_base
        }
        folds_gen_params = {"fold_num": fold_num, "seed": seed,
                            "target_col": self.target_col, "key_col": group_kfolds_col}

        self.runner = LgbmCvRunner(man_drop_cols, self.target_col, self.model_flag, folds_generator_flag,
                                   trainer_flag, self.evaluator_flag, trainer_params, model_params, folds_gen_params)

        self.seed = seed
        self.original_target_col = original_target_col

    def run(self, train, test):
        data = self.preprocess(train, test)
        output = self.runner.run(data)
        output["importance"] = self.aggregate_importances(output["importance"])
        return output

    def preprocess(self, org_train, org_test):
        """Adversarial validation用のデータセットを作る.train=0,test=1"""
        org_test = org_test.copy()
        org_train = org_train.drop(self.original_target_col, axis=1)
        org_train[self.target_col] = 0
        org_test[self.target_col] = 1
        concat_df = pd.concat([org_train, org_test])
        return concat_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def aggregate_importances(self, df):
        """feature importanceをfold平均する."""
        df["importance"] = df.mean(axis=1)
        return df[["importance"]]
