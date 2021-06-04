from lilac.ensemble.ensemble_runners.classifiers.multi import (
    AveragingMultiEnsemble, LgbmMultiEnsemble, LrMultiEnsemble)
from lilac.ensemble.ensemble_runners.regressors.rmse import (
    AveragingRmseEnsemble, LgbmRmseEnsemble, LinearRmseEnsemble,
    RandomForestRmseEnsemble, RidgeRmseEnsemble)
from lilac.ensemble.ensemble_runners.regressors.rmsle import (
    LgbmRmsleEnsemble, LinearRmsleEnsemble, RandomForestRmsleEnsemble,
    RidgeRmsleEnsemble)
from lilac.ensemble.ensemble_runners.regressors.mae import LgbmMaeEnsemble


class EnsembleRunnerFactory:
    def __init__(self, params):
        self.params = params
        self.required_params = ["target_col",
                                "folds_generator_flag", "folds_gen_params",
                                "trainer_flag", "trainer_params", "model_params"]

    def run(self, flag):
        # regressors
        # rmsle
        if flag == "lgbm_rmsle":
            Model = LgbmRmsleEnsemble
        elif flag == "linear_rmsle":
            Model = LinearRmsleEnsemble
        elif flag == "rf_rmsle":
            Model = RandomForestRmsleEnsemble
        elif flag == "ridge_rmsle":
            Model = RidgeRmsleEnsemble
        # rmse
        elif flag == "lgbm_rmse":
            Model = LgbmRmseEnsemble
        elif flag == "linear_rmse":
            Model = LinearRmseEnsemble
        elif flag == "rf_rmse":
            Model = RandomForestRmseEnsemble
        elif flag == "ridge_rmse":
            Model = RidgeRmseEnsemble
        elif flag == "avg_rmse":
            Model = AveragingRmseEnsemble
        # mae
        elif flag == "lgbm_mae":
            Model = LgbmMaeEnsemble
        # classfiers
        # multi
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
