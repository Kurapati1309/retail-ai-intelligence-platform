import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from src.utils.logging import get_logger

log = get_logger(__name__)

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

class TextEmbedder:
    def __init__(self, model_name="distilbert-base-uncased", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        log.info(f"Loaded embedder: {model_name} on {self.device}")

    @torch.no_grad()
    def encode(self, texts, batch_size=32, max_length=128) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks)
            pooled = mean_pool(out.last_hidden_state, toks["attention_mask"])
            vectors.append(pooled.cpu().numpy())
        return np.vstack(vectors)
