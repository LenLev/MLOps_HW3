import asyncio
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from app.embedding_onnx import OnnxEmbedder


@dataclass
class PendingRequest:
    texts: List[str]
    normalize: bool
    enqueued_at: float
    future: asyncio.Future


class DynamicBatcher:
    def __init__(
        self,
        embedder: OnnxEmbedder,
        max_wait_ms: int = 20,
        max_batch_requests: int = 32,
        max_batch_texts: int = 128,
    ) -> None:
        self.embedder = embedder
        self.max_wait_ms = max_wait_ms
        self.max_batch_requests = max_batch_requests
        self.max_batch_texts = max_batch_texts
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self.total_batches = 0
        self.total_texts = 0

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name="dynamic-batcher-worker")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def infer(self, texts: List[str], normalize: bool) -> tuple[List[List[float]], float]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        item = PendingRequest(
            texts=texts,
            normalize=normalize,
            enqueued_at=time.perf_counter(),
            future=future,
        )
        await self.queue.put(item)
        return await future

    async def _worker(self) -> None:
        while True:
            first = await self.queue.get()
            pending = [first]
            total_texts = len(first.texts)
            deadline = time.perf_counter() + (self.max_wait_ms / 1000.0)

            while len(pending) < self.max_batch_requests and total_texts < self.max_batch_texts:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break

                if total_texts + len(nxt.texts) > self.max_batch_texts:
                    await self.queue.put(nxt)
                    break

                pending.append(nxt)
                total_texts += len(nxt.texts)

            flattened: List[str] = []
            offsets: List[tuple[int, int]] = []
            for req in pending:
                start_idx = len(flattened)
                flattened.extend(req.texts)
                end_idx = len(flattened)
                offsets.append((start_idx, end_idx))

            batch_embeddings, _ = self.embedder.embed(flattened, normalize=False)
            self.total_batches += 1
            self.total_texts += len(flattened)

            for req, (start_idx, end_idx) in zip(pending, offsets):
                part = batch_embeddings[start_idx:end_idx]
                if req.normalize:
                    arr = np.array(part, dtype=np.float32)
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms = np.clip(norms, a_min=1e-12, a_max=None)
                    part = (arr / norms).tolist()
                total_latency_ms = (time.perf_counter() - req.enqueued_at) * 1000.0
                if not req.future.done():
                    req.future.set_result((part, total_latency_ms))
