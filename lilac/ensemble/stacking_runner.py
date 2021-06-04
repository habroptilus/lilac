from lilac.ensemble.ensemble_runner_factory import EnsembleRunnerFactory


class StackingRunner:
    """1段目の出力を受け取り,stackingを実行する."""

    def __init__(self, stackings, base_params):
        """stackings : どのモデルを使って何層stackingするか."""
        # key_colはgroup kfoldのときに使われる
        self.stackings = stackings
        params = {
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
            }
        }
        self.factory = EnsembleRunnerFactory(params)

    def run(self, output_list, train, test):
        if (len(self.stackings) == 0) and (len(output_list) > 1):
            raise Exception(
                f"1st layers output: {len(output_list)} but no stacking layers follows.")

        result_list = []
        input_list = output_list
        for i, layer in enumerate(self.stackings):
            layer_results = []
            for ensemble_flag in layer:
                runner = self.factory.run(ensemble_flag)
                result = runner.run(input_list, train, test)
                layer_results.append(result)
            result_list.append(layer_results)
            input_list = layer_results
        if len(input_list) != 1:
            raise Exception("Stacking output should be single.")

        return result_list
