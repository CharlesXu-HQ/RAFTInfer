# Benchmark methodology

The checked-in [Qwen3.5-9B BF16 RTX 5090 result](../benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl)
is accepted evidence, not a synthetic or cross-machine aggregate. It uses the
fixed Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16
GGUF SHA-256 `5e2d54b1b54df02cf1797e6a5e1465255ed68a9547bfd0ab0bde1357347d65e8`,
and llama.cpp revision `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`.

Each PP128, PP512, and TG128@PP512 arm has 5 warmups and 20 measurements. Before
benchmarking, exact greedy parity passes 4 records × 32 generated tokens (128
exact matches), with online-tiled attention, BF16 head-major KV cache, and
decode graph replay reported where applicable. The shared-GPU preflight runs
before parity and benchmarking to avoid publishing a contended measurement.

The chart's `RAFTInfer ratio` is `RAFTInfer tokens_per_second / llama.cpp
tokens_per_second` for the phase shown. Thus prefill displays pp128 and pp512,
while generation displays all three arms. A ratio above 1 means RAFTInfer's
throughput is higher. The `peak_allocated_gpu_bytes` field is RAFTInfer's RMM
logical allocation peak; it is not a process-wide GPU-memory measurement.

Limits: this is a Qwen3.5-9B BF16, single-RTX-5090, specified-protocol result.
It does not generalize to another model, quantization, GPU, batch size, service
latency, power, cost, or end-to-end serving workload. The only performance
comparison here is llama.cpp.
