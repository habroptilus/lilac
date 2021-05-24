from .regressors.lgbm_regressors import LgbmRmsleRegressor, LgbmRmseRegressor, LgbmMaeRegressor
from .regressors.linear_regressors import LinearRmsle, LinearModel, LinearPositiveModel, RidgeRmsle, RidgeModel
from .regressors.catb_regressors import CatbRmseRegressor, CatbRmsleRegressor
from .regressors.xgb_regressors import XgbRmseRegressor, XgbRmsleRegressor
from .regressors.random_forest_regressors import RandomForestRmseRegressor, RandomForestMaeRegressor, RandomForestRmsleRegressor
from .classifiers.lgbm_classifiers import LgbmBinaryClassifier, LgbmMultiClassifier
from .classifiers.logistic_regression import LrMultiClassifier
from .classifiers.catb_classifiers import CatbMultiClassifier


class ModelFactory:
    def __init__(self, model_str, params):
        self.params = params
        self.required_params = ["target_col"]
        # lgbm
        # regressor
        if model_str == "lgbm_rmsle":
            self.required_params.extend(
                ["verbose_eval", "num_boost_round", "early_stopping_rounds", "lgbm_params"])
            self.Model = LgbmRmsleRegressor
        elif model_str == "lgbm_rmse":
            self.required_params.extend(
                ["verbose_eval", "num_boost_round", "early_stopping_rounds", "lgbm_params"])
            self.Model = LgbmRmseRegressor
        elif model_str == "lgbm_mae":
            self.required_params.extend(
                ["verbose_eval", "num_boost_round", "early_stopping_rounds", "lgbm_params"])
            self.Model = LgbmMaeRegressor
        # classifier
        elif model_str == "lgbm_bin":
            self.required_params.extend(
                ["verbose_eval", "num_boost_round", "early_stopping_rounds", "lgbm_params"])
            self.Model = LgbmBinaryClassifier
        elif model_str == "lgbm_multi":
            self.required_params.extend(
                ["verbose_eval", "num_boost_round", "early_stopping_rounds", "lgbm_params", "num_class"])
            self.Model = LgbmMultiClassifier
        # xgb
        elif model_str == "xgb_rmse":
            self.required_params.extend(
                ["verbose_eval", "num_boost_round", "early_stopping_rounds", "xgb_params"])
            self.Model = XgbRmseRegressor
        elif model_str == "xgb_rmsle":
            self.required_params.extend(
                ["verbose_eval", "num_boost_round", "early_stopping_rounds", "xgb_params"])
            self.Model = XgbRmsleRegressor
        # catb
        elif model_str == "catb_rmse":
            self.required_params.extend(
                ["early_stopping_rounds", "catb_params"])
            self.Model = CatbRmseRegressor
        elif model_str == "catb_rmsle":
            self.required_params.extend(
                ["early_stopping_rounds", "catb_params"])
            self.Model = CatbRmsleRegressor
        elif model_str == "catb_multi":
            self.required_params.extend(
                ["early_stopping_rounds", "catb_params"])
            self.Model = CatbMultiClassifier
        # random forest
        elif model_str == "rf_rmse":
            self.required_params.append("seed")
            self.Model = RandomForestRmseRegressor
        elif model_str == "rf_rmsle":
            self.required_params.append("seed")
            self.Model = RandomForestRmsleRegressor
        elif model_str == "rf_mae":
            self.required_params.append("seed")
            self.Model = RandomForestMaeRegressor
        # linear
        # regressor
        elif model_str == "linear_rmsle":
            self.Model = LinearRmsle
        elif model_str == "linear_rmse":
            self.Model = LinearModel
        elif model_str == "linear_pos":
            self.Model = LinearPositiveModel
        elif model_str == "ridge_rmse":
            self.Model = RidgeModel
        elif model_str == "ridge_rmsle":
            self.Model = RidgeRmsle
        # classifier
        elif model_str == "lr_multi":
            self.Model = LrMultiClassifier
        else:
            raise Exception(f"Invalid model flag {model_str}")

    def run(self):
        params = {e: self.params[e]
                  for e in self.required_params}  # 必要なものだけ取り出す
        return self.Model(**params)
