# Architecture
**Retail AI Intelligence Platform (RAIIP)**

## Data sources
- Structured: transactions, customer profiles
- Unstructured: customer reviews (text)

## Pipeline
1. Synthetic data generation -> `data/raw`
2. Cleaning & standardization -> `data/processed`
3. Feature engineering:
   - Fraud: transaction-level features
   - Behavior: customer-level aggregates
   - NLP embeddings (optional extension)
4. Modeling:
   - Clustering: customer segmentation (KMeans)
   - Classification: fraud detection (LogReg baseline)
   - Regression: behavior prediction (Ridge baseline)
5. Evaluation:
   - Classification: accuracy/precision/recall/F1/ROC-AUC
   - Regression: MAE/RMSE
6. Serving:
   - FastAPI endpoints for fraud + behavior predictions

## Production extensions (easy upgrades)
- Swap baseline models for XGBoost/LightGBM
- Add time-window features (7/30/90 day)
- Add Kafka streaming ingestion
- Replace placeholder GenAI logic with your chosen LLM provider
