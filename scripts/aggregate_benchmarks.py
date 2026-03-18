import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def success_rate(data: dict[str, Any] | None) -> float | None:
    if not data:
        return None
    total = data.get("total_requests", 0)
    ok = data.get("successful_requests", 0)
    if total <= 0:
        return None
    return (ok / total) * 100.0


def row(label: str, data: dict[str, Any] | None) -> str:
    if not data:
        return f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"

    return (
        f"| {label}"
        f" | {fmt(data.get('throughput_rps'))}"
        f" | {fmt(data.get('latency_mean_ms'))}"
        f" | {fmt(data.get('latency_p95_ms'))}"
        f" | {fmt(data.get('latency_p99_ms'))}"
        f" | {fmt(data.get('cpu_percent_mean'))}"
        f" | {fmt(data.get('memory_mb_mean'))}"
        f" | {fmt(success_rate(data))} |"
    )


def delta_pct(cur: float | None, ref: float | None) -> str:
    if cur is None or ref is None or ref == 0:
        return "N/A"
    return f"{((cur - ref) / ref) * 100.0:+.2f}%"


def build_markdown(
    baseline: dict[str, Any] | None,
    onnx: dict[str, Any] | None,
    dynamic: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append("| Scenario | Throughput (RPS) | Mean Latency (ms) | P95 (ms) | P99 (ms) | CPU Mean (%) | RAM Mean (MB) | Success Rate (%) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(row("Baseline (transformers)", baseline))
    lines.append(row("ONNX Runtime", onnx))
    lines.append(row("ONNX + Dynamic Batch", dynamic))
    lines.append("")

    b_tp = baseline.get("throughput_rps") if baseline else None
    o_tp = onnx.get("throughput_rps") if onnx else None
    d_tp = dynamic.get("throughput_rps") if dynamic else None

    b_p95 = baseline.get("latency_p95_ms") if baseline else None
    o_p95 = onnx.get("latency_p95_ms") if onnx else None
    d_p95 = dynamic.get("latency_p95_ms") if dynamic else None

    lines.append("## Relative Changes")
    lines.append("")
    lines.append("| Comparison | Throughput Change | P95 Latency Change |")
    lines.append("|---|---:|---:|")
    lines.append(f"| ONNX vs Baseline | {delta_pct(o_tp, b_tp)} | {delta_pct(o_p95, b_p95)} |")
    lines.append(f"| Dynamic vs ONNX | {delta_pct(d_tp, o_tp)} | {delta_pct(d_p95, o_p95)} |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Positive throughput change is better.")
    lines.append("- Negative latency change is better.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark JSON files into a markdown table")
    parser.add_argument("--baseline", default="benchmark_results/baseline.json")
    parser.add_argument("--onnx", default="benchmark_results/onnx.json")
    parser.add_argument("--dynamic", default="benchmark_results/onnx_dynamic_batch.json")
    parser.add_argument("--out", default="benchmark_results/summary.md")
    args = parser.parse_args()

    baseline = load_json(Path(args.baseline))
    onnx = load_json(Path(args.onnx))
    dynamic = load_json(Path(args.dynamic))

    content = build_markdown(baseline, onnx, dynamic)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")

    print(content)
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
