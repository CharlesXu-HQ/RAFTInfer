# Blackwell RAFT Runtime

Blackwell RAFT Runtime is an RTX 50-series LLM inference runtime foundation. It
uses RAFT/RMM as the GPU resource and memory layer, then leaves
performance-critical operators to custom CUDA/CUTLASS kernels.

Current milestone: M2A Qwen3.5-9B loader foundation. The runtime now parses
GGUF v3 catalogs, derives the Qwen3.5 hybrid 3-linear/1-full block plan,
validates the text-model tensor manifest, memory-maps validated model files,
and exposes model/tokenizer metadata through coarse C++/Rust handles. It does
not yet upload weights or run Qwen3.5 inference.

## Scope

- Targets RTX 50-series consumer Blackwell only. v0.1 does not support RTX 40
  series.
- C++ owns GPU pointers, streams, events, allocations, graphs, RAFT resources,
  RMM resources, and custom kernel launches.
- Rust uses coarse FFI calls into opaque C++ engine handles. It does not call
  individual GPU operators directly.
- M0 imports no `bw24` source code. Future `bw24` custom kernels may be reused
  directly only when they are already performance-optimized and after license,
  provenance, functional/numerical correctness, and target-performance evidence
  are recorded.

## Host checks

```bash
scripts/local-check.sh
```

This runs the host-only CMake build, CTest, Rust formatting check, and Rust
workspace tests with the CUDA backend disabled.

## RTX 50 GPU smoke

The target-verified image build used explicit regional endpoints because the
official Rustup endpoint stalled on the validation host:

```bash
docker build --progress=plain \
  --build-arg CMAKE_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg RUSTUP_DIST_SERVER=https://rsproxy.cn \
  --build-arg RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup \
  -f containers/Dockerfile.dev \
  -t brt-dev:26.06-cuda13 .
scripts/gpu-smoke.sh
```

The Dockerfile defaults remain the official PyPI and Rust endpoints. Use the
build arguments only when the target environment requires an explicit mirror.

The GPU runner refuses to start when it detects another compute workload,
insufficient memory headroom, excessive utilization, invalid GPU telemetry, or
another active BRT smoke run. It exposes only GPU 0 to the container and repeats
the preflight inside the container before launching the smoke kernel.

The verified M0 golden output is:

```json
{"device_id":0,"element_count":1024,"checksum":523776}
```

See [docs/m1-verification.md](docs/m1-verification.md) for the current M1
verification contract, [docs/m2a-verification.md](docs/m2a-verification.md) for
the host-only Qwen3.5 loader evidence,
[docs/m0-verification.md](docs/m0-verification.md) for the original target
smoke evidence, and
[docs/provenance/dependencies.md](docs/provenance/dependencies.md) for
dependency provenance.
