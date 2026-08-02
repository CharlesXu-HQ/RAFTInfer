# Checked-in benchmark results

`results/qwen35-9b-bf16-rtx5090.jsonl` is the accepted schema-v2 source for
the README visual. Records are committed only after pinned provenance, exact
parity, and performance-floor gates pass. Do not edit measured fields to make a
chart look better; publish a new accepted evidence record instead.

When more than one independently stable run passes, commit the conservative
representative: a run whose displayed ratios are not higher than the confirming
run. Retain the confirmation report and its hash in release verification notes.

Render the asset with the Python standard-library renderer:

```bash
python3 tools/render_benchmark_chart.py benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl docs/assets/qwen35-bf16-rtx5090.svg
tests/benchmark-asset-test.sh
```

The renderer accepts exactly `pp128`, `pp512`, and `tg128_pp512`, schema 2,
passing parity, and passing performance floors. It emits a deterministic static
1200×620 SVG with no host paths or timestamps.
