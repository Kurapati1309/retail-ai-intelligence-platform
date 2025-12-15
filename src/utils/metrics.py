from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error
)
import numpy as np

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None

def classification_metrics(y_true, y_pred, y_prob=None) -> ClassificationMetrics:
    roc = None
    if y_prob is not None:
        try:
            roc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc = None
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc
    )

@dataclass
class RegressionMetrics:
    mae: float
    rmse: float

def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return RegressionMetrics(mae=mae, rmse=rmse)
