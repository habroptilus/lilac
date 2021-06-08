import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_squared_log_error, roc_auc_score, accuracy_score, f1_score)


class EvaluatorBase:
    def __init__(self, flag):
        self.flag = flag

    def run(self, y, pred, raw_pred):
        y = np.array(y)
        pred = np.array(pred)
        raw_pred = np.array(raw_pred)
        including_null_rows = np.isnan(pred)

        if including_null_rows.sum() > 0:
            print("[Waring] prediction have nan. So drop before evaluation.")
            pred = pred[~including_null_rows]
            raw_pred = raw_pred[~including_null_rows]
            y = y[~including_null_rows]
        return self._run(y, pred, raw_pred)

    def _run(self, df):
        raise Exception("Implement please.")

    def return_flag(self):
        return self.flag

    def get_direction(self):
        # 基本minimizeなので、maximizeのときは書き換える
        return "minimize"


class RmsleEvaluator(EvaluatorBase):
    """RMSLEで評価する."""

    def _run(self,  y, pred, raw_pred):
        return np.sqrt(mean_squared_log_error(y, pred))


class RmseEvaluator(EvaluatorBase):
    """RMSEで評価する."""

    def _run(self,  y, pred, raw_pred):
        return np.sqrt(mean_squared_error(y, pred))


class MaeEvaluator(EvaluatorBase):
    """MAEで評価する."""

    def _run(self,  y, pred, raw_pred):
        return mean_absolute_error(y, pred)


class AucEvaluator(EvaluatorBase):
    """AUCで評価する."""

    def _run(self,  y, pred, raw_pred):
        return roc_auc_score(y, raw_pred)

    def get_direction(self):
        return "maximize"


class AccuracyEvaluator(EvaluatorBase):
    """Accuracyで評価する."""

    def _run(self,  y, pred, raw_pred):
        return accuracy_score(y, pred)

    def get_direction(self):
        return "maximize"


class MacroF1Evaluator(EvaluatorBase):
    """macro f1_scoreで評価する."""

    def _run(self, y, pred, raw_pred):
        return f1_score(y, pred, average='macro')

    def get_direction(self):
        return "maximize"
