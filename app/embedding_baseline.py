import time
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class BaselineEmbedder:
    def __init__(self, model_name: str = "sergeyzh/rubert-mini-frida") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: List[str], normalize: bool = True) -> tuple[List[List[float]], float]:
        start = time.perf_counter()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

        arr = embeddings.cpu().numpy().astype(np.float32)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            arr = arr / norms

        latency_ms = (time.perf_counter() - start) * 1000.0
        return arr.tolist(), latency_ms
