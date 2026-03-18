import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class OnnxEmbedder:
    def __init__(
        self,
        model_name: str = "sergeyzh/rubert-mini-frida",
        onnx_path: str = "models/onnx/model.onnx",
    ) -> None:
        self.model_name = model_name
        self.onnx_path = str(Path(onnx_path))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        masked = last_hidden_state * mask
        summed = masked.sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    def embed(self, texts: List[str], normalize: bool = True) -> tuple[List[List[float]], float]:
        start = time.perf_counter()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np",
        )

        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

        outputs = self.session.run(None, inputs)
        last_hidden_state = outputs[0].astype(np.float32)
        pooled = self._mean_pool(last_hidden_state, inputs["attention_mask"])

        if normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            pooled = pooled / norms

        latency_ms = (time.perf_counter() - start) * 1000.0
        return pooled.tolist(), latency_ms
