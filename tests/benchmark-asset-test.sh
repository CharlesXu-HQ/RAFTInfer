#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary="$(mktemp "${TMPDIR:-/tmp}/raftinfer-benchmark-chart.XXXXXX")"
trap 'rm -f "${temporary}"' EXIT

python3 "${repo_root}/tools/render_benchmark_chart.py" \
  "${repo_root}/benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl" "${temporary}"
cmp "${temporary}" "${repo_root}/docs/assets/qwen35-bf16-rtx5090.svg"
for expected in 6495.48 8409.16 87.47 86.91 86.25 1.889x 1.190x 1.032x 1.027x 1.019x; do
  grep -Fq "${expected}" "${temporary}"
done
grep -Fq 'text-anchor="end">6495.48 tok/s' "${temporary}"
grep -Fq 'text-anchor="start">3438.78 tok/s' "${temporary}"
