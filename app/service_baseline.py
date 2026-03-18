import os

from fastapi import FastAPI

from app.embedding_baseline import BaselineEmbedder
from app.schemas import EmbedRequest, EmbedResponse, HealthResponse


MODEL_NAME = os.getenv("MODEL_NAME", "sergeyzh/rubert-mini-frida")

app = FastAPI(title="Baseline Embedding Service", version="1.0.0")
embedder = BaselineEmbedder(model_name=MODEL_NAME)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", backend="transformers", model_name=MODEL_NAME)


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest) -> EmbedResponse:
    embeddings, latency_ms = embedder.embed(payload.texts, normalize=payload.normalize)
    return EmbedResponse(
        embeddings=embeddings,
        model_name=MODEL_NAME,
        backend="transformers",
        batch_size=len(payload.texts),
        latency_ms=latency_ms,
    )
