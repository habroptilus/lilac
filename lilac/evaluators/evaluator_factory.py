from .evaluator_base import RmsleEvaluator, RmseEvaluator, MaeEvaluator, AucEvaluator, AccuracyEvaluator, MacroF1Evaluator


class EvaluatorFactory:
    def run(self, flag):
        if flag == "rmsle":
            return RmsleEvaluator(flag)
        elif flag == "rmse":
            return RmseEvaluator(flag)
        elif flag == "mae":
            return MaeEvaluator(flag)
        elif flag == "auc":
            return AucEvaluator(flag)
        elif flag == "accuracy":
            return AccuracyEvaluator(flag)
        elif flag == "f1_macro":
            return MacroF1Evaluator(flag)
        else:
            raise Exception(f"Invalid flag : {flag}")
