import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import httpx
import psutil


DEFAULT_TEXTS = [
    "Машинное обучение помогает автоматизировать рутинные задачи.",
    "FastAPI позволяет быстро собирать HTTP-сервисы на Python.",
    "ONNX Runtime часто ускоряет инференс на CPU.",
    "Динамическое батчирование увеличивает пропускную способность при высокой нагрузке.",
    "Latency и throughput важно измерять вместе.",
]


@dataclass
class BenchmarkResult:
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_s: float
    throughput_rps: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cpu_percent_mean: float | None
    memory_mb_mean: float | None


def percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    idx = int((len(values) - 1) * q)
    return sorted(values)[idx]


async def worker(
    client: httpx.AsyncClient,
    url: str,
    requests_count: int,
    texts_per_request: int,
    latencies: List[float],
    failures: List[int],
    error_messages: List[str],
) -> None:
    for _ in range(requests_count):
        payload = {
            "texts": random.choices(DEFAULT_TEXTS, k=texts_per_request),
            "normalize": True,
        }
        start = time.perf_counter()
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            _ = response.json()
            latencies.append((time.perf_counter() - start) * 1000.0)
        except Exception as exc:
            failures.append(1)
            if len(error_messages) < 10:
                error_messages.append(repr(exc))


async def sample_resources(pid: int, stop_event: asyncio.Event, values: List[tuple[float, float]]) -> None:
    process = psutil.Process(pid)
    process.cpu_percent(interval=None)
    while not stop_event.is_set():
        await asyncio.sleep(0.2)
        try:
            cpu = process.cpu_percent(interval=None)
            mem = process.memory_info().rss / (1024 * 1024)
            values.append((cpu, mem))
        except psutil.Error:
            break


async def run_benchmark(
    name: str,
    url: str,
    total_requests: int,
    concurrency: int,
    texts_per_request: int,
    server_pid: int | None,
) -> BenchmarkResult:
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=60.0)
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency)

    latencies: List[float] = []
    failures: List[int] = []
    error_messages: List[str] = []
    resource_samples: List[tuple[float, float]] = []
    stop_event = asyncio.Event()

    base = total_requests // concurrency
    rem = total_requests % concurrency
    per_worker = [base + (1 if i < rem else 0) for i in range(concurrency)]

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
        tasks = [
            asyncio.create_task(
                worker(client, url, cnt, texts_per_request, latencies, failures, error_messages)
            )
            for cnt in per_worker
            if cnt > 0
        ]

        sampler_task = None
        if server_pid is not None:
            sampler_task = asyncio.create_task(sample_resources(server_pid, stop_event, resource_samples))

        await asyncio.gather(*tasks)
        stop_event.set()
        if sampler_task is not None:
            await sampler_task

    total_time_s = time.perf_counter() - start
    successful_requests = len(latencies)
    failed_requests = len(failures)

    cpu_mean = None
    mem_mean = None
    if resource_samples:
        cpu_mean = statistics.mean(x[0] for x in resource_samples)
        mem_mean = statistics.mean(x[1] for x in resource_samples)

    return BenchmarkResult(
        name=name,
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        total_time_s=total_time_s,
        throughput_rps=(successful_requests / total_time_s) if total_time_s > 0 else 0.0,
        latency_mean_ms=statistics.mean(latencies) if latencies else math.nan,
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        latency_p99_ms=percentile(latencies, 0.99),
        cpu_percent_mean=cpu_mean,
        memory_mb_mean=mem_mean,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP benchmark for embedding services")
    parser.add_argument("--name", required=True, help="Benchmark scenario name")
    parser.add_argument("--url", required=True, help="Embedding endpoint URL")
    parser.add_argument("--total-requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--texts-per-request", type=int, default=1)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--out", default="benchmark_results/result.json")
    args = parser.parse_args()

    result = asyncio.run(
        run_benchmark(
            name=args.name,
            url=args.url,
            total_requests=args.total_requests,
            concurrency=args.concurrency,
            texts_per_request=args.texts_per_request,
            server_pid=args.server_pid,
        )
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
