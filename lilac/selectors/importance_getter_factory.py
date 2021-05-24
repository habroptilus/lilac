from lilac.selectors.lgbm_importance_getters import MeanFeatureImportanceGetter, NullImportanceGetter


class ImportanceGetterFactory:
    def __init__(self, flag):
        self.required_params = ["man_drop_cols", "target_col", "model_flag", "folds_generator_flag",
                                "trainer_flag", "evaluator_flag", "trainer_params", "model_params",
                                "folds_gen_params"]

        if flag == "mean":
            self.Model = MeanFeatureImportanceGetter
        elif flag == "null_importance":
            self.required_params.append("n_trials")
            self.Model = NullImportanceGetter

    def run(self, params):
        params = {e: params[e]
                  for e in self.required_params}  # 必要なものだけ取り出す
        return self.Model(**params)
