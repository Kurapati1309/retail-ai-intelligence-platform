import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

PRODUCT_CATS = ["Lumber", "Plumbing", "Electrical", "Paint", "Tools", "Garden", "Appliances"]
CHANNELS = ["Store", "Online", "Pickup"]
PAYMENT = ["Card", "Wallet", "GiftCard"]

REVIEW_TEMPLATES = [
    "Product quality is {adj}, delivery was {adj2}.",
    "I {verb} this item. It is {adj} and {adj2}.",
    "Customer service was {adj}, the item arrived {adj2}.",
    "Not satisfied: {reason}. Would {verb2} again? {yn}.",
    "Great value for money, but packaging was {adj}.",
]

ADJ = ["excellent", "good", "average", "poor", "terrible"]
ADJ2 = ["fast", "on time", "late", "damaged", "missing"]
VERB = ["love", "like", "hate", "recommend", "won't recommend"]
REASON = ["damaged on arrival", "missing parts", "wrong item", "poor build quality", "late delivery"]
YN = ["yes", "no"]

def make_review():
    t = RNG.choice(REVIEW_TEMPLATES)
    return t.format(
        adj=RNG.choice(ADJ),
        adj2=RNG.choice(ADJ2),
        verb=RNG.choice(VERB),
        reason=RNG.choice(REASON),
        verb2=RNG.choice(["buy", "purchase", "order"]),
        yn=RNG.choice(YN),
    )

def generate(n_customers=5000, n_txns=200000, out_dir="data/raw"):
    os.makedirs(out_dir, exist_ok=True)

    customers = pd.DataFrame({
        "customer_id": [f"C{str(i).zfill(6)}" for i in range(n_customers)],
        "tenure_months": RNG.integers(1, 120, size=n_customers),
        "region": RNG.choice(["South", "Midwest", "West", "Northeast"], size=n_customers),
        "is_pro": RNG.choice([0, 1], size=n_customers, p=[0.78, 0.22]),
    })

    txns = pd.DataFrame({
        "txn_id": [f"T{str(i).zfill(8)}" for i in range(n_txns)],
        "customer_id": RNG.choice(customers["customer_id"], size=n_txns),
        "channel": RNG.choice(CHANNELS, size=n_txns, p=[0.55, 0.35, 0.10]),
        "product_category": RNG.choice(PRODUCT_CATS, size=n_txns),
        "units": RNG.integers(1, 8, size=n_txns),
        "unit_price": np.round(RNG.uniform(2, 450, size=n_txns), 2),
        "payment_type": RNG.choice(PAYMENT, size=n_txns, p=[0.75, 0.20, 0.05]),
        "has_coupon": RNG.choice([0, 1], size=n_txns, p=[0.7, 0.3]),
        "return_flag": RNG.choice([0, 1], size=n_txns, p=[0.92, 0.08]),
    })

    base = np.datetime64("2025-01-01")
    txns["txn_ts"] = base + RNG.integers(0, 365, size=n_txns).astype("timedelta64[D]")
    txns["amount"] = np.round(txns["units"] * txns["unit_price"] * (1 - 0.05 * txns["has_coupon"]), 2)

    fraud_prob = (
        0.002
        + 0.006 * (txns["amount"] > 800)
        + 0.010 * (txns["payment_type"] == "GiftCard").astype(int)
        + 0.008 * txns["return_flag"]
        + 0.004 * (txns["channel"] == "Online").astype(int)
    )
    txns["fraud_label"] = (RNG.random(n_txns) < fraud_prob).astype(int)

    reviewed = txns.sample(frac=0.12, random_state=42)[["txn_id", "customer_id", "product_category"]].copy()
    reviewed["review_text"] = [make_review() for _ in range(len(reviewed))]

    customers.to_csv(os.path.join(out_dir, "customers.csv"), index=False)
    txns.to_csv(os.path.join(out_dir, "transactions.csv"), index=False)
    reviewed.to_csv(os.path.join(out_dir, "reviews.csv"), index=False)

    print("Wrote:", out_dir)

if __name__ == "__main__":
    generate()
