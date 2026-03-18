import os

from fastapi import FastAPI

from app.batching import DynamicBatcher
from app.embedding_onnx import OnnxEmbedder
from app.schemas import EmbedRequest, EmbedResponse, HealthResponse


MODEL_NAME = os.getenv("MODEL_NAME", "sergeyzh/rubert-mini-frida")
ONNX_PATH = os.getenv("ONNX_PATH", "models/onnx/model.onnx")
MAX_WAIT_MS = int(os.getenv("MAX_WAIT_MS", "10"))
MAX_BATCH_REQUESTS = int(os.getenv("MAX_BATCH_REQUESTS", "24"))
MAX_BATCH_TEXTS = int(os.getenv("MAX_BATCH_TEXTS", "96"))

app = FastAPI(title="Dynamic Batch ONNX Service", version="1.0.0")
embedder = OnnxEmbedder(model_name=MODEL_NAME, onnx_path=ONNX_PATH)
batcher = DynamicBatcher(
    embedder=embedder,
    max_wait_ms=MAX_WAIT_MS,
    max_batch_requests=MAX_BATCH_REQUESTS,
    max_batch_texts=MAX_BATCH_TEXTS,
)


@app.on_event("startup")
async def on_startup() -> None:
    await batcher.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await batcher.stop()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", backend="onnxruntime+dynamic_batch", model_name=MODEL_NAME)


@app.get("/batch_stats")
def batch_stats() -> dict:
    return {
        "max_wait_ms": MAX_WAIT_MS,
        "max_batch_requests": MAX_BATCH_REQUESTS,
        "max_batch_texts": MAX_BATCH_TEXTS,
        "total_batches": batcher.total_batches,
        "total_texts": batcher.total_texts,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(payload: EmbedRequest) -> EmbedResponse:
    embeddings, latency_ms = await batcher.infer(payload.texts, normalize=payload.normalize)
    return EmbedResponse(
        embeddings=embeddings,
        model_name=MODEL_NAME,
        backend="onnxruntime+dynamic_batch",
        batch_size=len(payload.texts),
        latency_ms=latency_ms,
    )
