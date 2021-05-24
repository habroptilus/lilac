from .evaluator_base import RmsleEvaluator, RmseEvaluator, MaeEvaluator, AucEvaluator, AccuracyEvaluator, MacroF1Evaluator


class EvaluatorFactory:
    def __init__(self, target_col):
        self.target_col = target_col

    def run(self, flag):
        if flag == "rmsle":
            return RmsleEvaluator(flag, self.target_col)
        elif flag == "rmse":
            return RmseEvaluator(flag, self.target_col)
        elif flag == "mae":
            return MaeEvaluator(flag, self.target_col)
        elif flag == "auc":
            return AucEvaluator(flag, self.target_col)
        elif flag == "accuracy":
            return AccuracyEvaluator(flag, self.target_col)
        elif flag == "f1_macro":
            return MacroF1Evaluator(flag, self.target_col)
        else:
            raise Exception(f"Invalid flag : {flag}")
