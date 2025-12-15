from fastapi import FastAPI
from src.api.schemas import FraudRequest, FraudResponse, BehaviorRequest, BehaviorResponse
from src.api.service import ModelService

app = FastAPI(title="RAIIP API", version="1.0.0")
svc = ModelService()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/fraud", response_model=FraudResponse)
def predict_fraud(req: FraudRequest):
    label, prob = svc.predict_fraud(req.model_dump())
    return FraudResponse(fraud_label=label, fraud_probability=prob)

@app.post("/predict/behavior", response_model=BehaviorResponse)
def predict_behavior(req: BehaviorRequest):
    pred = svc.predict_behavior(req.model_dump())
    return BehaviorResponse(predicted_spend=pred)
