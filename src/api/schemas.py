from pydantic import BaseModel
from typing import Optional

class FraudRequest(BaseModel):
    channel: str
    product_category: str
    payment_type: str
    units: int
    unit_price: float
    amount: float
    has_coupon: int
    return_flag: int
    is_online: int
    is_giftcard: int
    log_amount: float

class FraudResponse(BaseModel):
    fraud_label: int
    fraud_probability: Optional[float] = None

class BehaviorRequest(BaseModel):
    txn_count: float
    total_spend: float
    avg_spend: float
    return_rate: float
    high_value_rate: float

class BehaviorResponse(BaseModel):
    predicted_spend: float
