from lilac.validators.folds_generators import (FoldsGenerator,
                                               GroupKFoldsGenerator,
                                               StratifiedFoldsGenerator,
                                               StratifiedGroupKFold,
                                               TimeSeriesKfold,
                                               TimeSeriesSimpleKfold)


class FoldsGeneratorFactory:
    def run(self, flag, params):
        required_params = ["fold_num"]
        if flag == "kfold":
            required_params.append("seed")
            Model = FoldsGenerator
        elif flag == "stratified":
            required_params.extend(["target_col", "seed"])
            Model = StratifiedFoldsGenerator
        elif flag == "group":
            required_params.extend(["key_col", "seed"])
            Model = GroupKFoldsGenerator
        elif flag == "stratified_group":
            required_params.extend(["target_col", "key_col", "seed"])
            Model = StratifiedGroupKFold
        elif flag == "time_series":
            required_params.extend(["ts_column", "clipping"])
            Model = TimeSeriesKfold
        elif flag == "time_series_simple":
            required_params.extend(["ts_column"])
            Model = TimeSeriesSimpleKfold
        else:
            raise Exception(f"Invalid flag :{flag}")
        necesary_params = {e: params[e]
                           for e in required_params}
        return Model(**necesary_params)
