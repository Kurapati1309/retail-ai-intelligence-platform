import os
import joblib
import numpy as np
import pandas as pd
from src.config import PATHS

def load_model(name: str):
    path = os.path.join(PATHS.model_dir, name)
    return joblib.load(path)

class ModelService:
    def __init__(self):
        self.fraud_model = None
        self.behavior_model = None

    def ensure_loaded(self):
        if self.fraud_model is None:
            self.fraud_model = load_model("fraud_model.joblib")
        if self.behavior_model is None:
            self.behavior_model = load_model("behavior_model.joblib")

    def predict_fraud(self, payload: dict):
        self.ensure_loaded()
        X = pd.DataFrame([payload])
        pred = int(self.fraud_model.predict(X)[0])
        prob = None
        try:
            prob = float(self.fraud_model.predict_proba(X)[:, 1][0])
        except Exception:
            prob = None
        return pred, prob

    def predict_behavior(self, payload: dict):
        self.ensure_loaded()
        X = pd.DataFrame([payload])
        pred = float(self.behavior_model.predict(X)[0])
        return pred
