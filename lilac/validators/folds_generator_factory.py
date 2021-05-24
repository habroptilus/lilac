from lilac.validators.folds_generators import StratifiedFoldsGenerator, FoldsGenerator, GroupKFoldsGenerator, StratifiedGroupKFold


class FoldsGeneratorFactory:
    def run(self, flag, params):
        required_params = ["fold_num", "seed"]
        if flag == "kfold":
            Model = FoldsGenerator
        elif flag == "stratified":
            required_params.append("target_col")
            Model = StratifiedFoldsGenerator
        elif flag == "group":
            required_params.append("key_col")
            Model = GroupKFoldsGenerator
        elif flag == "stratified_group":
            required_params.extend(["target_col", "key_col"])
            Model = StratifiedGroupKFold
        else:
            raise Exception(f"Invalid flag :{flag}")
        necesary_params = {e: params[e]
                           for e in required_params}
        return Model(**necesary_params)
