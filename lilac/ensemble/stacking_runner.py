from lilac.ensemble.ensemble_runner_factory import EnsembleRunnerFactory
from pathlib import Path
import pandas as pd


class StackingRunner:
    """1段目の出力を受け取り,stackingを実行する."""

    def __init__(self, stackings, base_params, use_original_features):
        """
        :params:
        :stackings : どのモデルを使って何層stackingするか.
        :use_original_features
        """
        self.stackings = stackings
        self.params = {
            "target_col": base_params["target_col"],
            "folds_generator_flag": base_params["folds_generator_flag"],
            "folds_gen_params": {"fold_num":  base_params["fold_num"],
                                 "seed":  base_params["seed"],
                                 "key_col":  base_params["group_kfolds_col"],
                                 "target_col": base_params["target_col"],
                                 "ts_column": base_params["ts_column"],
                                 "clipping": base_params["clipping"]
                                 },
            "trainer_flag":  base_params["trainer_flag"],
            "trainer_params": {
                "target_col": base_params["target_col"],
                "bagging_num": base_params["bagging_num"],
                "base_class": base_params["base_class"],
                "seed": base_params["seed"],
                "allow_less_than_base": base_params["allow_less_than_base"]
            },
            "model_params": {
                "target_col": base_params["target_col"],
                "verbose_eval": base_params["verbose_eval"],
                "early_stopping_rounds": base_params["early_stopping_rounds"],
                "class_weight": base_params["class_weight"],
                "seed": base_params["seed"],
                "lgbm_params": {
                    "colsample_bytree": base_params["colsample_bytree"],
                    "max_depth": base_params["max_depth"],
                    "reg_alpha": base_params["reg_alpha"],
                    "reg_lambda": base_params["reg_lambda"],
                    "subsample": base_params["subsample"],
                    "min_child_weight": base_params["min_child_weight"],
                    "num_leaves": int(2 ** (base_params["max_depth"]) * 0.7),
                    "n_estimators": base_params["n_estimators"],
                    "random_state": base_params["seed"]
                }
            },
            "drop_cols": base_params["drop_cols"],
            "use_original_features": use_original_features
        }
        self.base_params = base_params
        self.factory = EnsembleRunnerFactory()

    def run(self, output_list):
        if (len(self.stackings) == 0) and (len(output_list) > 1):
            raise Exception(
                f"1st layers output: {len(output_list)} but no stacking layers follows.")

        result_list = []
        input_list = output_list
        for i, layer in enumerate(self.stackings):
            layer_results = []
            for ensemble_dict in layer:
                ensemble_flag = ensemble_dict["model"]
                ensemble_params = ensemble_dict.get("params")
                if ensemble_params:
                    # self.paramsを更新したくなかったのでこの書き方
                    updated_params = {**self.params, **ensemble_params}
                else:
                    updated_params = self.params.copy()

                runner = self.factory.run(ensemble_flag, updated_params)
                feature_dir = Path(
                    f"{self.base_params['result_dir']}/{self.base_params['luigi_dir']}/{self.base_params['features_conf_key']}")
                train = pd.read_csv(feature_dir/"train.csv")
                test = pd.read_csv(feature_dir/"test.csv")

                result = runner.run(input_list, train, test)
                layer_results.append(result)
            result_list.append(layer_results)
            input_list = layer_results
        if len(input_list) != 1:
            raise Exception("Stacking output should be single.")

        return result_list
