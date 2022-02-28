"""settings templateを作成するclass."""
import json


class Settings:
    """とりあえずそのまま出力する."""
    settings = {
        "default": {
            "result_dir": "result",
            "features_dir": "features",
            "luigi_dir": "luigi",
            "target_col": "y",
            "evaluator_flag": "rmse",
            "feature_importance_model_flag": "lgbm_rmse",
            "features_conf_key": "nothing",
            "model_flag": "lgbm_rmse",
            "importance_flag": "mean",
            "null_importance_trials": 10,
            "trainer_flag": "basic",
            "folds_generator_flag": "kfold",
            "group_kfolds_col": "gameID",
            "verbose_eval": 100,
            "n_estimators": 2000,
            "class_weight": "balanced",
            "early_stopping_rounds": 20,
            "fold_num": 5,
            "seed": 42,
            "bagging_num": 5,
            "base_class": 3,
            "allow_less_than_base": True,
            "drop_cols": [],
            "threshold": None,
            "colsample_bytree": 0.8,
            "max_depth": 5,
            "reg_alpha": 0,
            "reg_lambda": 0,
            "subsample": 0.8,
            "min_child_weight": 1.0,
            "lr": 0.1,
            "random_strength": 1,
            "bagging_temperature": 0.1,
            "od_type": "IncToDec",
            "od_wait": 10,
            "fs_colsample_bytree": 0.8,
            "fs_max_depth": 5,
            "fs_reg_alpha": 0,
            "fs_reg_lambda": 0,
            "fs_subsample": 0.8,
            "fs_min_child_weight": 1.0,
            "ts_column": "datetime",
            "clipping": False,
            "use_original_features": False
        },
        "run": {
            "sample": {
                "members": [
                    {
                        "features_conf_key": "nothing",
                        "model_flag": "lgbm_rmse"
                    }
                ],
                "stacking_key": "single"
            }
        },
        "features": {
            "nothing": []
        },
        "stackings": {
            "single": {
                "layers": []
            },
            "sample": {
                "layers": [
                    [
                        {
                            "model": "linear_rmse"
                        },
                        {
                            "model": "lgbm_rmse",
                            "params": {
                                "features_conf_key": "v1_0_0",
                                "use_original_features": True
                            }
                        }
                    ],
                    [
                        {
                            "model": "avg_rmse"
                        }
                    ]
                ],
                "params": {}
            },
            "avg": {
                "layers": [
                    [
                        {
                            "model": "avg_rmse"
                        }
                    ]
                ],
                "params": {}
            }
        },
        "template": {}
    }

    def __init__(self, output_path):
        self.output_path = output_path

    def run(self):
        with open(self.output_path, "w") as f:
            json.dump(self.settings, f, indent=4)
