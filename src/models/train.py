import os
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PATHS
from src.utils.io import read_csv
from src.utils.logging import get_logger
from src.models.fraud_classifier import build_fraud_model
from src.models.behavior_regressor import build_behavior_model
from src.models.clustering import build_customer_segmentation
from src.models.evaluate import eval_classifier, eval_regressor

log = get_logger(__name__)

def train_fraud():
    df = read_csv(os.path.join(PATHS.data_processed, "fraud_features.csv"))
    y = df["fraud_label"].astype(int)
    X = df.drop(columns=["fraud_label", "txn_id", "customer_id", "txn_ts"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_fraud_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    m = eval_classifier(y_test, y_pred, y_prob)
    log.info(f"Fraud metrics: {m}")

    os.makedirs(PATHS.model_dir, exist_ok=True)
    out = os.path.join(PATHS.model_dir, "fraud_model.joblib")
    joblib.dump(model, out)
    log.info(f"Saved: {out}")

def train_behavior():
    df = read_csv(os.path.join(PATHS.data_processed, "behavior_features.csv"))
    y = df["target_spend"].astype(float)
    X = df.drop(columns=["target_spend", "customer_id"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_behavior_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    m = eval_regressor(y_test, y_pred)
    log.info(f"Behavior metrics: {m}")

    os.makedirs(PATHS.model_dir, exist_ok=True)
    out = os.path.join(PATHS.model_dir, "behavior_model.joblib")
    joblib.dump(model, out)
    log.info(f"Saved: {out}")

def train_cluster():
    df = read_csv(os.path.join(PATHS.data_processed, "behavior_features.csv"))
    feature_cols = ["txn_count", "total_spend", "avg_spend", "return_rate", "high_value_rate"]
    model = build_customer_segmentation(n_clusters=5)
    model.fit(df[feature_cols])

    os.makedirs(PATHS.model_dir, exist_ok=True)
    out = os.path.join(PATHS.model_dir, "cluster_model.joblib")
    joblib.dump(model, out)
    log.info(f"Saved: {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["fraud", "behavior", "cluster"])
    args = parser.parse_args()

    if args.model == "fraud":
        train_fraud()
    elif args.model == "behavior":
        train_behavior()
    else:
        train_cluster()

if __name__ == "__main__":
    main()
