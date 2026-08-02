#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary="$(mktemp "${TMPDIR:-/tmp}/raftinfer-benchmark-chart.XXXXXX")"
trap 'rm -f "${temporary}"' EXIT

python3 "${repo_root}/tools/render_benchmark_chart.py" \
  "${repo_root}/benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl" "${temporary}"
cmp "${temporary}" "${repo_root}/docs/assets/qwen35-bf16-rtx5090.svg"
for expected in 6489.06 8440.00 87.46 85.07 84.81 1.896x 1.193x 1.031x 1.004x 1.001x; do
  grep -Fq "${expected}" "${temporary}"
done
grep -Fq 'text-anchor="end">6489.06 tok/s' "${temporary}"
grep -Fq 'text-anchor="start">3422.78 tok/s' "${temporary}"
