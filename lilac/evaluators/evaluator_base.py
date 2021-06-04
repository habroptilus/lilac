import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_squared_log_error, roc_auc_score, accuracy_score, f1_score)


class EvaluatorBase:
    def __init__(self, flag, target_col):
        self.flag = flag
        self.target_col = target_col

    def run(self, df, pred, raw_pred):
        df = df.copy()
        df["pred"] = pred
        df["raw_pred"] = raw_pred
        if len(df["pred"].isnull()) or len(df["raw_pred"].isnull()):
            print("[Waring] prediction have nan. So drop and evaluate.")
            df = df.dropna()
        return self._run(df[[self.target_col, "pred", "raw_pred"]])

    def _run(self, df):
        raise Exception("Implement please.")

    def return_flag(self):
        return self.flag

    def get_direction(self):
        # 基本minimizeなので、maximizeのときは書き換える
        return "minimize"


class RmsleEvaluator(EvaluatorBase):
    """RMSLEで評価する."""

    def _run(self, df):
        return np.sqrt(mean_squared_log_error(df[self.target_col], df["pred"]))


class RmseEvaluator(EvaluatorBase):
    """RMSEで評価する."""

    def _run(self, df):
        return np.sqrt(mean_squared_error(df[self.target_col], df["pred"]))


class MaeEvaluator(EvaluatorBase):
    """MAEで評価する."""

    def _run(self, df):
        return mean_absolute_error(df[self.target_col], df["pred"])


class AucEvaluator(EvaluatorBase):
    """AUCで評価する."""

    def _run(self, df):
        return roc_auc_score(df[self.target_col], df["raw_pred"])

    def get_direction(self):
        return "maximize"


class AccuracyEvaluator(EvaluatorBase):
    """Accuracyで評価する."""

    def _run(self, df):
        return accuracy_score(df[self.target_col], df["pred"])

    def get_direction(self):
        return "maximize"


class MacroF1Evaluator(EvaluatorBase):
    """macro f1_scoreで評価する."""

    def _run(self, df):
        return f1_score(df[self.target_col], df["pred"], average='macro')

    def get_direction(self):
        return "maximize"
