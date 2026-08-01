#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary="$(mktemp "${TMPDIR:-/tmp}/raftinfer-benchmark-chart.XXXXXX")"
trap 'rm -f "${temporary}"' EXIT

python3 "${repo_root}/tools/render_benchmark_chart.py" \
  "${repo_root}/benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl" "${temporary}"
cmp "${temporary}" "${repo_root}/docs/assets/qwen35-bf16-rtx5090.svg"
for expected in 6491.86 8391.24 87.46 85.01 84.98 1.967x 1.284x 1.038x 1.010x 1.009x; do
  grep -Fq "${expected}" "${temporary}"
done
grep -Fq 'text-anchor="end">6491.86 tok/s' "${temporary}"
grep -Fq 'text-anchor="start">3300.20 tok/s' "${temporary}"
