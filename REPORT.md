# Отчёт по HW3

## Что запускалось

Сравнение 3 вариантов одного и того же сервиса эмбеддингов на CPU:

1. Обычный инференс через transformers
2. Инференс через onnxruntime после конвертации модели в ONNX
3. ONNX + динамическое батчирование (очередь + worker)

Модель: sergeyzh/rubert-mini-frida

Окружение:

- ОС: Windows
- CPU: AMD Ryzen 5 3600 (6/12)
- Python: 3.10.11

Нагрузка:

- total_requests: 300
- texts_per_request: 1
- concurrency: 20 для baseline и ONNX, 50 для dynamic batch

## Какие метрики и зачем

- throughput_rps: сколько запросов в секунду обрабатывает сервис.
- latency_p50_ms, latency_p95_ms, latency_p99_ms: типичная и «хвостовая» задержка.
- cpu_percent_mean, memory_mb_mean: потребление ресурсов (в этом прогоне не снималось).

## Результаты

### 1) Baseline (transformers)

| Метрика | Значение |
|---|---:|
| throughput_rps | 90.14 |
| latency_mean_ms | 181.90 |
| latency_p50_ms | 100.41 |
| latency_p95_ms | 547.25 |
| latency_p99_ms | 1333.26 |
| cpu_percent_mean | N/A |
| memory_mb_mean | N/A |

Коротко: всё работает стабильно, но скорость средняя.

### 2) ONNX Runtime

| Метрика | Значение |
|---|---:|
| throughput_rps | 110.74 |
| latency_mean_ms | 141.67 |
| latency_p50_ms | 65.02 |
| latency_p95_ms | 530.81 |
| latency_p99_ms | 1155.84 |
| cpu_percent_mean | N/A |
| memory_mb_mean | N/A |

Сравнение с baseline:

- throughput: +22%
- p95 latency: -3%

ONNX оказался лучше baseline по скорости и задержке.

### 3) ONNX + Dynamic Batch

Параметры батчинга:

- MAX_WAIT_MS=10
- MAX_BATCH_REQUESTS=24
- MAX_BATCH_TEXTS=96

| Метрика | Значение |
|---|---:|
| throughput_rps | 58.41 |
| latency_mean_ms | 647.88 |
| latency_p50_ms | 309.09 |
| latency_p95_ms | 2434.86 |
| latency_p99_ms | 3475.82 |
| cpu_percent_mean | N/A |
| memory_mb_mean | N/A |

Сравнение с ONNX без батчинга:

- throughput: -47%
- p95 latency: `+358.7%

для этой нагрузки и этих параметров динамический батчинг сделал хуже.

## Общий вывод

В данном эксперименте лучший вариант - ONNX Runtime без динамического батчинга

- Лучший по latency - ONNX Runtime.
- Лучший по throughput: ONNX Runtime.
- По CPU/RAM вывод сделать нельзя, потому что эти метрики не снимались в этом прогоне.

Если нужен именно выигрыш от dynamic batch, надо отдельно тюнить параметры и профиль нагрузки (больше параллельности, другое окно ожидания и размер батча).
