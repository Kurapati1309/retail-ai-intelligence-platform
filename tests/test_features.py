import pandas as pd
from src.features.build_features import build_fraud_features

def test_build_fraud_features_adds_columns():
    df = pd.DataFrame([{
        "txn_id":"T00000001","customer_id":"C000001","channel":"Online","product_category":"Tools",
        "payment_type":"GiftCard","units":2,"unit_price":10.0,"amount":20.0,"has_coupon":0,"return_flag":1
    }])
    out = build_fraud_features(df)
    assert "is_online" in out.columns
    assert "is_giftcard" in out.columns
    assert "log_amount" in out.columns
