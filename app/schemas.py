from typing import List

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of input texts")
    normalize: bool = Field(default=True, description="L2 normalize output embeddings")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    backend: str
    batch_size: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_name: str
