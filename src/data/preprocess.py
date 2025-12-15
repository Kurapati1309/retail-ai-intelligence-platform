import os
import pandas as pd
from src.config import PATHS
from src.utils.io import read_csv, write_csv
from src.utils.logging import get_logger
from src.utils.text import basic_clean

log = get_logger(__name__)

def preprocess():
    os.makedirs(PATHS.data_processed, exist_ok=True)

    customers = read_csv(os.path.join(PATHS.data_raw, "customers.csv"))
    txns = read_csv(os.path.join(PATHS.data_raw, "transactions.csv"))
    reviews = read_csv(os.path.join(PATHS.data_raw, "reviews.csv"))

    # Basic hygiene
    txns["txn_ts"] = pd.to_datetime(txns["txn_ts"])
    txns = txns.drop_duplicates("txn_id")

    reviews["review_text"] = reviews["review_text"].fillna("").map(basic_clean)

    write_csv(customers, os.path.join(PATHS.data_processed, "customers_clean.csv"))
    write_csv(txns, os.path.join(PATHS.data_processed, "transactions_clean.csv"))
    write_csv(reviews, os.path.join(PATHS.data_processed, "reviews_clean.csv"))

    log.info("Preprocess complete ✅")

if __name__ == "__main__":
    preprocess()
