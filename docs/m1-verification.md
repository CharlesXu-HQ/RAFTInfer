# M1 Verification

## Host-only

- Command: `scripts/local-check.sh`
- Expected result: PASS
- CUDA backend: disabled
- Coverage: host CMake configure/build, CTest, `cargo fmt --check`, and Rust
  workspace tests.
- Required CTest coverage: tensor validation, workspace layout, deterministic
  operator registry dispatch, independent reference operators, correctness
  metrics, C ABI source contracts, CUDA source contracts, and benchmark JSONL
  serialization.

## Benchmark JSONL Evidence

Every benchmark record is one newline-terminated JSON object with deterministic
field order and `schema_version: 1`. String fields must escape quotes,
backslashes, and every control byte below `0x20`; project-native kernels record
`upstream_revision: null`.

Required M1 evidence fields:

- registry identity: operator name, selected kernel name, backend, optional
  upstream revision, target architecture, dtype, and shape bucket;
- correctness state: pass/fail, maximum absolute error, maximum relative error,
  cosine similarity, and non-finite mismatch count;
- timing state: measured iterations, median microseconds, p95 microseconds, and
  the derived `performance_publishable` flag.

Metrics and timings must be finite before serialization. Non-finite values make
the record invalid rather than publishable evidence.

## Correctness-valid vs. Publishable Performance

A correctness-valid record proves that the candidate output passed its
independent CPU oracle and tolerance checks. It can be used to debug operator
behavior even when no timing was collected.

A publishable performance record requires all of the following:

- correctness passed;
- measured iterations are greater than zero;
- median latency is finite and positive;
- p95 latency is finite and positive.

Performance data cannot bypass correctness. A record with failed correctness is
not publishable even when iterations and timing fields look valid.

## Target GPU

Target validation still uses the fail-closed shared-GPU preflight before any
CUDA work:

```bash
docker build --progress=plain \
  --build-arg CMAKE_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg RUSTUP_DIST_SERVER=https://rsproxy.cn \
  --build-arg RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup \
  -f containers/Dockerfile.dev \
  -t brt-dev:26.06-cuda13 .
scripts/gpu-smoke.sh
```

The target evidence should capture GPU model, compute capability, driver,
CUDA/toolchain versions, image ID, preflight state, the exact command, and the
observed smoke output.

## Scope Not Covered

M1 does not load Qwen3 Dense, parse GGUF, tokenize prompts, allocate a KV cache,
execute a Transformer layer, capture or replay a CUDA Graph, validate inference
quality, or publish end-to-end model latency.
