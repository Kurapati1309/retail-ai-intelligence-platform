# Retail AI Intelligence Platform (RAIIP)
JD-aligned, experience-level project for a **Data Engineer / AI-ML Engineer** role (Python + NLP + LLM/GenAI + experiments + model evaluation + deployment).

## What this repo demonstrates
- **Experiment design** to validate hypotheses and model assumptions (A/B-style comparisons, metric-driven decisions)
- **Clustering / Classification / Regression** pipelines
- **NLP + embeddings** from unstructured text (customer reviews)
- **Model evaluation** (accuracy, precision, recall, F1, ROC-AUC, RMSE/MAE)
- **Production-style packaging** (src layout, logging, config, tests)
- **API service** (FastAPI) to serve predictions and insights

## Quickstart
### 1) Create env + install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Generate realistic retail data (structured + unstructured)
```bash
python src/data/generate_synthetic.py
```
Outputs:
- `data/raw/customers.csv`
- `data/raw/transactions.csv`
- `data/raw/reviews.csv`

### 3) Preprocess + build features
```bash
python -m src.data.preprocess
python -m src.features.build_features
```

### 4) Train + evaluate models
```bash
python -m src.models.train --model fraud
python -m src.models.train --model behavior
python -m src.models.train --model cluster
```

### 5) Run API
```bash
uvicorn src.api.main:app --reload
```
Then open:
- `http://127.0.0.1:8000/docs`

## Repo structure
See `docs/architecture.md` for the end-to-end architecture and `docs/experiment_plan.md` for the experiment plan.

## Notes
- Default deep learning framework is **PyTorch** (also compatible with “TensorFlow/PyTorch” requirement).
- LLM/GenAI is demonstrated via **review summarization + insight generation** (placeholder local logic). Swap with your preferred LLM provider later.
