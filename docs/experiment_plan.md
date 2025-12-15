# Experiment Plan (JD-aligned)

## Hypotheses
1. Adding engineered features (log_amount, return_rate) increases fraud precision.
2. Adding review sentiment/embeddings improves behavior prediction accuracy.

## Experiment design
- Baseline vs Variant comparison (A/B style)
- Fixed train/test split with stratification for fraud label
- Metrics tracked:
  - Fraud: precision/recall/F1/ROC-AUC
  - Behavior: MAE/RMSE
- Decision rule:
  - Fraud: prefer higher precision at minimum recall threshold (business constraint)
  - Behavior: prefer lower RMSE

## Tracking
- Use `src/experiments/mlflow_tracking.py` to enable MLflow tracking.
