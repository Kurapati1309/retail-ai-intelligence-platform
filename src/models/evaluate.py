import numpy as np
from src.utils.metrics import classification_metrics, regression_metrics

def eval_classifier(y_true, y_pred, y_prob=None):
    return classification_metrics(y_true, y_pred, y_prob)

def eval_regressor(y_true, y_pred):
    return regression_metrics(y_true, y_pred)
