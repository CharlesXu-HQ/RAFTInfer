# Task 6 report — RAFTInfer documentation and benchmark visual

## TDD evidence

`tests/benchmark-chart-test.py` was created before the renderer. Its initial
run was RED because `tools/render_benchmark_chart.py` did not exist. The test
then passed GREEN after the renderer was implemented; it runs a real subprocess
against a hand-written schema-v2 JSONL fixture and tests valid rendering plus
schema, arm, parity, and performance-floor refusals.

## Chart contract and data

The SVG is deterministic, static, and 1200×620. It has vertical grouped bars
in Prefill (2 categories) and Generation (3 categories) panels, zero baselines,
direct series names, numeric labels, and source-ratio labels. RAFTInfer is
`#76B900`; llama.cpp is `#64748B`.

Formal input SHA-256:
`181f1a7539f75eec495efd2b66e701123443273a53f6c704bfe5b62141954259`.
Parity source SHA-256:
`628826717bd85a227bd5921062ec067338627ec97c54aaafd4b64ede968597ee`.
The checked-in records only migrate schema 1→2 and RAFTInfer brand keys; all
measured, provenance, parity, execution, GPU, software, latency, throughput,
variation, memory, and ratio evidence is retained.

## Verification

- `python3 tests/benchmark-chart-test.py`
- `python3 tools/render_benchmark_chart.py benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl docs/assets/qwen35-bf16-rtx5090.svg`
- `tests/benchmark-asset-test.sh`
- `python3 tests/readme-links-test.py`
- `tests/public-surface-test.sh`
- `scripts/check-project-brand.sh`
- `scripts/local-check.sh`

## Integration fix

The published provenance path is now the portable
`<artifact-root>/Qwen3.5-9B-GGUF/Qwen3.5-9B-c202236-bf16.provenance.json`, and
`peak_memory_status` is `measured_by_raftinfer_rmm`. These are non-measurement
metadata and brand portability changes. An independent normalizer reads the
immutable formal source and the checked-in result, permits only schema-version,
brand-key/object, portable-path, and memory-status changes, and reports:

```text
benchmark-normalization: pass records=3 allowed_changes=schema,brand,portable-path,memory-status
```

After the owning tasks converted their negative fixtures to runtime input
construction, `scripts/check-project-brand.sh` and the complete
`scripts/local-check.sh` are green.

## Visual QA fix

The renderer now places every ratio strictly above its group’s highest bar with
a 28-pixel vertical margin, keeping ratios outside the painted bars. Paired
throughput labels use outward anchors (`end` for RAFTInfer and `start` for
llama.cpp), so adjacent labels have separate horizontal intervals. The new
renderer behavior test was RED against the former layout (`205.0` was inside a
bar whose top was `185.0`) and is GREEN after the layout change. The asset test
also locks the outward-anchor output. A Quick Look PNG render was visually
reviewed for unclipped panels, labels, zero baselines, and separated ratios.

## Concerns

The accepted comparison is limited to the recorded Qwen3.5-9B BF16 RTX 5090
protocol and llama.cpp. It is not a `bw24` comparison or a claim about other
hardware, models, workloads, latency, power, or cost.
