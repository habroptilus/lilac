import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_squared_log_error, roc_auc_score, accuracy_score, f1_score)


class EvaluatorBase:
    def __init__(self, flag, target_col):
        self.flag = flag
        self.target_col = target_col

    def run(self, df, pred, raw_pred):
        raise Exception("Implement please.")

    def return_flag(self):
        return self.flag

    def get_direction(self):
        # 基本minimizeなので、maximizeのときは書き換える
        return "minimize"


class RmsleEvaluator(EvaluatorBase):
    """RMSLEで評価する."""

    def run(self, df, pred, raw_pred):
        y = df[self.target_col]
        return np.sqrt(mean_squared_log_error(y, pred))


class RmseEvaluator(EvaluatorBase):
    """RMSEで評価する."""

    def run(self, df, pred, raw_pred):
        y = df[self.target_col]
        pred = pred, raw_pred.predict(df)
        return np.sqrt(mean_squared_error(y, pred))


class MaeEvaluator(EvaluatorBase):
    """MAEで評価する."""

    def run(self, df, pred, raw_pred):
        y = df[self.target_col]
        return mean_absolute_error(y, pred)


class AucEvaluator(EvaluatorBase):
    """AUCで評価する."""

    def run(self, df, pred, raw_pred):
        y = df[self.target_col]
        return roc_auc_score(y, raw_pred)

    def get_direction(self):
        return "maximize"


class AccuracyEvaluator(EvaluatorBase):
    """Accuracyで評価する."""

    def run(self, df, pred, raw_pred):
        y = df[self.target_col]
        return accuracy_score(y, pred)

    def get_direction(self):
        return "maximize"


class MacroF1Evaluator(EvaluatorBase):
    """macro f1_scoreで評価する."""

    def run(self, df, pred, raw_pred):
        y = df[self.target_col]
        return f1_score(y, pred, average='macro')

    def get_direction(self):
        return "maximize"
