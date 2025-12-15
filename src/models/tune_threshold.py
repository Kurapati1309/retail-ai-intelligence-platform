import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PATHS
from src.utils.io import read_csv
from src.utils.logging import get_logger
from src.models.fraud_classifier import build_fraud_model
from src.utils.metrics import classification_metrics

log = get_logger(__name__)

def tune_threshold(min_recall: float = 0.30):
    # Load features
    df = read_csv(os.path.join(PATHS.data_processed, "fraud_features.csv"))

    # Split labels/features
    y = df["fraud_label"].astype(int)
    X = df.drop(columns=["fraud_label", "txn_id", "customer_id", "txn_ts"], errors="ignore")

    # Train/test split with stratification (important for imbalanced fraud)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = build_fraud_model()
    model.fit(X_train, y_train)

    # Predicted probabilities for fraud
    probs = model.predict_proba(X_test)[:, 1]

    # Try thresholds
    thresholds = np.linspace(0.05, 0.95, 19)

    best = None
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        m = classification_metrics(y_test, preds, probs)
        results.append((t, m.precision, m.recall, m.f1, m.roc_auc))

        # Business rule: keep at least min_recall, maximize precision
        if m.recall >= min_recall:
            if best is None or m.precision > best["precision"]:
                best = {"threshold": float(t), "precision": m.precision, "recall": m.recall,
                        "f1": m.f1, "roc_auc": m.roc_auc}

    # Print a small table (top 10 by precision)
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    log.info("Top thresholds by precision (threshold, precision, recall, f1, roc_auc):")
    for row in results_sorted:
        log.info(row)

    if best is None:
        log.info(f"No threshold met min_recall={min_recall}. Try lower min_recall (ex: 0.20).")
    else:
        log.info(f"BEST threshold under recall constraint (min_recall={min_recall}): {best}")

if __name__ == "__main__":
    tune_threshold(min_recall=0.30)
