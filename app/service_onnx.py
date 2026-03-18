import os

from fastapi import FastAPI

from app.embedding_onnx import OnnxEmbedder
from app.schemas import EmbedRequest, EmbedResponse, HealthResponse


MODEL_NAME = os.getenv("MODEL_NAME", "sergeyzh/rubert-mini-frida")
ONNX_PATH = os.getenv("ONNX_PATH", "models/onnx/model.onnx")

app = FastAPI(title="ONNX Embedding Service", version="1.0.0")
embedder = OnnxEmbedder(model_name=MODEL_NAME, onnx_path=ONNX_PATH)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", backend="onnxruntime", model_name=MODEL_NAME)


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest) -> EmbedResponse:
    embeddings, latency_ms = embedder.embed(payload.texts, normalize=payload.normalize)
    return EmbedResponse(
        embeddings=embeddings,
        model_name=MODEL_NAME,
        backend="onnxruntime",
        batch_size=len(payload.texts),
        latency_ms=latency_ms,
    )
