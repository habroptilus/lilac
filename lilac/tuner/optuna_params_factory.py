class OptunaParamsFactory:
    @classmethod
    def get_params(cls, trial):
        return {'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1e3),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1e3),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 0.95),
                'subsample': trial.suggest_uniform('subsample', 0.6, 0.95),
                'max_depth': trial.suggest_int('max_depth', low=3, high=9),
                'min_child_weight': trial.suggest_loguniform('min_child_weight', low=.1, high=10),
                'lr': trial.suggest_loguniform('lr', 0.01, 0.3),
                'random_strength': trial.suggest_int('random_strength', 0, 100),
                'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
                'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                'od_wait': trial.suggest_int('od_wait', 10, 50),


                }

    @classmethod
    def run(cls, model_flag, trial):
        if model_flag.startswith("lgbm_") or model_flag.startswith("xgb_"):
            return {'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1e3),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1e3),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 0.95),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 0.95),
                    'max_depth': trial.suggest_int('max_depth', low=3, high=9),
                    'min_child_weight': trial.suggest_loguniform('min_child_weight', low=.1, high=10)}
        elif model_flag.startswith("catb_"):
            return {'lr': trial.suggest_loguniform('lr', 0.01, 0.3),
                    'random_strength': trial.suggest_int('random_strength', 0, 100),
                    'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
                    'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                    'od_wait': trial.suggest_int('od_wait', 10, 50)}
        elif model_flag == "fs":
            return {'fs_reg_lambda': trial.suggest_loguniform('fs_reg_lambda', 1e-3, 1e3),
                    'fs_reg_alpha': trial.suggest_loguniform('fs_reg_alpha', 1e-3, 1e3),
                    'fs_colsample_bytree': trial.suggest_uniform('fs_colsample_bytree', .6, 0.95),
                    'fs_subsample': trial.suggest_uniform('fs_subsample', .6, 0.95),
                    'fs_max_depth': trial.suggest_int('fs_max_depth', low=3, high=9),
                    'fs_min_child_weight': trial.suggest_loguniform('fs_min_child_weight', low=.1, high=10)}
        elif model_flag == "threshold":
            return {"threshold": trial.suggest_uniform('threshold', 0.1, 1.0)}
        else:
            return []
