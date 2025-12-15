import os
import pandas as pd
from src.config import PATHS
from src.utils.io import read_csv, write_csv
from src.utils.logging import get_logger

log = get_logger(__name__)

def build_behavior_features(txns: pd.DataFrame) -> pd.DataFrame:
    # Customer-level aggregations (rolling windows could be added later)
    grp = txns.groupby("customer_id").agg(
        txn_count=("txn_id", "count"),
        total_spend=("amount", "sum"),
        avg_spend=("amount", "mean"),
        return_rate=("return_flag", "mean"),
        high_value_rate=("amount", lambda x: (x > 800).mean()),
    ).reset_index()

    # Behavior label proxy: next-month spend (simplified). Here we predict total_spend as demonstration.
    grp["target_spend"] = grp["total_spend"]
    return grp

def build_fraud_features(txns: pd.DataFrame) -> pd.DataFrame:
    # Transaction-level features for fraud classification
    feats = txns.copy()
    feats["is_online"] = (feats["channel"] == "Online").astype(int)
    feats["is_giftcard"] = (feats["payment_type"] == "GiftCard").astype(int)
    feats["log_amount"] = (feats["amount"].clip(lower=0.01)).apply(lambda v: __import__("math").log(v))
    return feats

def main():
    txns = read_csv(os.path.join(PATHS.data_processed, "transactions_clean.csv"))
    behavior = build_behavior_features(txns)
    fraud = build_fraud_features(txns)

    write_csv(behavior, os.path.join(PATHS.data_processed, "behavior_features.csv"))
    write_csv(fraud, os.path.join(PATHS.data_processed, "fraud_features.csv"))
    log.info("Feature build complete ✅")

if __name__ == "__main__":
    main()
