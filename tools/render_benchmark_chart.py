#!/usr/bin/env python3
"""Render checked-in Qwen3.5 benchmark JSONL as a deterministic SVG."""

import json
from pathlib import Path
import sys


WIDTH = 1200
HEIGHT = 620
RAFTINFER = "#76B900"
LLAMA_CPP = "#64748B"
ARMS = ("pp128", "pp512", "tg128_pp512")


class ChartError(ValueError):
    pass


def fail(message):
    raise ChartError(f"benchmark-chart: {message}")


def load_records(path):
    try:
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        fail(f"unable to read JSONL: {error}")
    by_arm = {}
    for record in records:
        arm = record.get("arm")
        if record.get("schema_version") != 2:
            fail(f"arm {arm!r} must have schema_version 2")
        if arm in by_arm:
            fail(f"duplicate arm {arm!r}")
        by_arm[arm] = record
    if len(records) != 3 or set(by_arm) != set(ARMS):
        fail("arms must be exactly pp128, pp512, and tg128_pp512")
    for arm in ARMS:
        record = by_arm[arm]
        if record.get("parity", {}).get("passed") is not True:
            fail(f"arm {arm} did not pass parity")
        if record.get("performance_floor_passed") is not True:
            fail(f"arm {arm} did not pass the performance-floor")
        for implementation in ("raftinfer", "llama_cpp"):
            for phase in ("prefill", "generation"):
                value = record.get(implementation, {}).get(phase, {}).get("tokens_per_second")
                if not isinstance(value, (int, float)) or value <= 0:
                    fail(f"arm {arm} has invalid {implementation}.{phase}.tokens_per_second")
        for phase in ("prefill", "generation"):
            ratio = record.get("throughput_ratio", {}).get(phase)
            if not isinstance(ratio, (int, float)) or ratio <= 0:
                fail(f"arm {arm} has invalid throughput_ratio.{phase}")
    return by_arm


def escape(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, content, size=14, anchor="start", weight="normal", fill="#0F172A"):
    return (f'  <text x="{x}" y="{y}" fill="{fill}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(content)}</text>')


def bar(x, baseline, height, color, label, label_y):
    return [
        f'  <rect x="{x:.2f}" y="{baseline - height:.2f}" width="46" height="{height:.2f}" fill="{color}"/>',
        text(f"{x + 23:.2f}", f"{label_y:.2f}", label, 12, "middle", "bold"),
    ]


def panel(lines, x, width, title, categories, records, phase):
    # Reserve a distinct title band above all plotted value labels.
    top, baseline = 185, 510
    lines.extend([
        f'  <rect x="{x}" y="105" width="{width}" height="445" fill="#F8FAFC" stroke="#CBD5E1"/>',
        text(x + 24, 139, title, 22, "start", "bold"),
        text(x + width - 24, 139, "throughput (tok/s)", 12, "end", "normal", "#475569"),
        f'  <line x1="{x + 28}" y1="{baseline}" x2="{x + width - 28}" y2="{baseline}" stroke="#0F172A" stroke-width="2"/>',
    ])
    values = [
        records[arm][implementation][phase]["tokens_per_second"]
        for arm in categories for implementation in ("raftinfer", "llama_cpp")
    ]
    maximum = max(values)
    available_height = baseline - top
    slot = (width - 80) / len(categories)
    for index, arm in enumerate(categories):
        record = records[arm]
        center = x + 40 + slot * (index + 0.5)
        raftinfer_value = record["raftinfer"][phase]["tokens_per_second"]
        llama_value = record["llama_cpp"][phase]["tokens_per_second"]
        raftinfer_height = available_height * raftinfer_value / maximum
        llama_height = available_height * llama_value / maximum
        lines.extend(bar(center - 50, baseline, raftinfer_height, RAFTINFER,
                         f"{raftinfer_value:.2f} tok/s", baseline - raftinfer_height - 8))
        lines.extend(bar(center + 4, baseline, llama_height, LLAMA_CPP,
                         f"{llama_value:.2f} tok/s", baseline - llama_height - 8))
        ratio = record["throughput_ratio"][phase]
        ratio_y = max(top + 20, min(baseline - max(raftinfer_height, llama_height) - 29, baseline - 24))
        lines.append(text(f"{center:.2f}", f"{ratio_y:.2f}", f"RAFTInfer {ratio:.3f}x", 13,
                          "middle", "bold", "#14532D"))
        lines.append(text(f"{center:.2f}", 534, arm.replace("_", " "), 13, "middle", "bold"))


def render(records):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
        '  <title id="title">Qwen3.5-9B BF16 throughput on NVIDIA GeForce RTX 5090</title>',
        '  <desc id="description">RAFTInfer and llama.cpp throughput comparison. Values are tokens per second; each group includes the RAFTInfer to llama.cpp ratio.</desc>',
        '  <rect width="1200" height="620" fill="#FFFFFF"/>',
        text(40, 46, "Qwen3.5-9B BF16 — RTX 5090 throughput", 28, "start", "bold"),
        text(40, 72, "Accepted evidence: RAFTInfer vs llama.cpp; higher is better", 14, "start", "normal", "#475569"),
        f'  <rect x="850" y="56" width="14" height="14" fill="{RAFTINFER}"/>',
        text(871, 68, "RAFTInfer", 13),
        f'  <rect x="969" y="56" width="14" height="14" fill="{LLAMA_CPP}"/>',
        text(990, 68, "llama.cpp", 13),
    ]
    panel(lines, 40, 500, "Prefill", ("pp128", "pp512"), records, "prefill")
    panel(lines, 660, 500, "Generation", ARMS, records, "generation")
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def main(argv):
    if len(argv) != 3:
        print("usage: render_benchmark_chart.py INPUT_JSONL OUTPUT_SVG", file=sys.stderr)
        return 2
    try:
        output = Path(argv[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render(load_records(argv[1])), encoding="utf-8")
    except ChartError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
