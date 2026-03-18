# HW3: Оптимизация инференса

Модель: sergeyzh/rubert-mini-frida

В проекте есть 3 варианта сервиса:

1. Baseline: transformers на CPU.
2. ONNX: onnxruntime на CPU.
3. ONNX + динамическое батчирование.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## запуск

### 1) Baseline

```bash
uvicorn app.service_baseline:app --host 127.0.0.1 --port 8000
```

Бенчмарк:

```bash
python scripts/benchmark_http.py --name baseline --url http://127.0.0.1:8000/embed --total-requests 300 --concurrency 20 --texts-per-request 1 --out benchmark_results/baseline.json
```

### 2) ONNX

Сначала экспорт модели:

```bash
python scripts/export_to_onnx.py
```

Потом запуск сервиса:

```bash
uvicorn app.service_onnx:app --host 127.0.0.1 --port 8001
```

Бенчмарк:

```bash
python scripts/benchmark_http.py --name onnx --url http://127.0.0.1:8001/embed --total-requests 300 --concurrency 20 --texts-per-request 1 --out benchmark_results/onnx.json
```

### 3) ONNX + Dynamic Batch

```bash
set MAX_WAIT_MS=10
set MAX_BATCH_REQUESTS=24
set MAX_BATCH_TEXTS=96
uvicorn app.service_dynamic_batch:app --host 127.0.0.1 --port 8002
```

Бенчмарк:

```bash
python scripts/benchmark_http.py --name onnx_dynamic_batch --url http://127.0.0.1:8002/embed --total-requests 300 --concurrency 50 --texts-per-request 1 --out benchmark_results/onnx_dynamic_batch.json
```

## Сводная таблица

После трёх прогонов соберите сводку:

```bash
python scripts/aggregate_benchmarks.py --baseline benchmark_results/baseline.json --onnx benchmark_results/onnx.json --dynamic benchmark_results/onnx_dynamic_batch.json --out benchmark_results/summary.md
```

## Что измеряется

- `latency_mean_ms`, `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`
- `throughput_rps`
- `cpu_percent_mean`, `memory_mb_mean` (если передать PID сервиса)

Пример с PID:

```bash
python scripts/benchmark_http.py --name baseline --url http://127.0.0.1:8000/embed --server-pid 12345 --out benchmark_results/baseline.json
```

## Где смотреть результаты

- JSON-файлы: папка benchmark_results/
- Краткая сводка: benchmark_results/summary.md
- Отчёт: REPORT.md
