from lilac.ensemble.ensemble_runners import (LgbmMultiEnsemble,
                                             LgbmRmsleEnsemble,
                                             LinearRmsleEnsemble,
                                             LrMultiEnsemble,
                                             RandomForestRmsleEnsemble,
                                             RidgeRmsleEnsemble,
                                             AveragingMultiEnsemble)


class EnsembleRunnerFactory:
    def __init__(self, params):
        self.params = params
        self.required_params = ["target_col",
                                "folds_generator_flag", "folds_gen_params",
                                "trainer_flag", "trainer_params"]

    def run(self, flag):
        if flag == "lgbm_rmsle":
            Model = LgbmRmsleEnsemble
        elif flag == "linear_rmsle":
            Model = LinearRmsleEnsemble
        elif flag == "rf_rmsle":
            Model = RandomForestRmsleEnsemble
        elif flag == "ridge_rmsle":
            Model = RidgeRmsleEnsemble
        elif flag == "lgbm_multi":
            Model = LgbmMultiEnsemble
        elif flag == "lr_multi":
            Model = LrMultiEnsemble
        elif flag == "avg_multi":
            Model = AveragingMultiEnsemble
        else:
            raise Exception(f"Invalid ensemble flag: {flag}")

        params = {e: self.params[e]
                  for e in self.required_params}  # 必要なものだけ取り出す
        return Model(**params)
