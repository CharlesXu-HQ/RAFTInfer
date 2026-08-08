# Benchmark methodology

The checked-in [Qwen3.5-9B BF16 RTX 5090 result](../benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl)
is accepted evidence, not a synthetic or cross-machine aggregate. It uses the
fixed Qwen3.5-9B ModelScope revision `460979c3d11864dd16408d860ac930a360a2fac2`, BF16
GGUF SHA-256 `5e2d54b1b54df02cf1797e6a5e1465255ed68a9547bfd0ab0bde1357347d65e8`,
schema-v2 provenance SHA-256
`a7e1730316d25f8ad878afddbc76d6cb150c3ed191327423efd49b1df687109c`, and
llama.cpp revision `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`.

Each PP128, PP512, and TG128@PP512 arm has 5 warmups and 20 measurements. Before
benchmarking, exact greedy parity passes 4 records × 32 generated tokens (128
exact matches), with online-tiled attention, BF16 head-major KV cache, and
decode graph replay reported where applicable. The shared-GPU preflight runs
before parity and benchmarking to avoid publishing a contended measurement.

The accepted `auto` policy is the measured `split-k-256` default. PP128 remains
on the zero-workspace single-block path. PP512 and TG128@PP512 disclose a
256-token partition and threshold, a 1,024-token context bucket, 264,192 bytes
of fixed attention workspace, and captured split-K graph replay.

The chart's `RAFTInfer ratio` is `RAFTInfer tokens_per_second / llama.cpp
tokens_per_second` for the phase shown. Thus prefill displays pp128 and pp512,
while generation displays all three arms. A ratio above 1 means RAFTInfer's
throughput is higher. The `peak_allocated_gpu_bytes` field is RAFTInfer's RMM
logical allocation peak; it is not a process-wide GPU-memory measurement.

Two independently preflighted, uncontended split-K runs met both the BF16 and
promotion gates. The checked-in record is formal sample A, the conservative
representative for the promotion-critical long-context generation arms: its
PP512 and TG128@PP512 baseline-relative gains were both lower than sample B's.
This avoids publishing the faster of the two qualifying long-context samples.
All displayed values are direct RAFTInfer-to-llama.cpp ratios from that retained
sample; no `bw24` series is included.

Limits: this is a Qwen3.5-9B BF16, single-RTX-5090, specified-protocol result.
It does not generalize to another model, quantization, GPU, batch size, service
latency, power, cost, or end-to-end serving workload. The only performance
comparison here is llama.cpp.
