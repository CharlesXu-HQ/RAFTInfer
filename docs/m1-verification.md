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

Required M1 evidence fields, in stable JSON order:

- `schema_version`: currently `1`;
- `utc_timestamp`: UTC collection timestamp;
- `project_commit`: project revision that produced the record;
- `device`: GPU/device name;
- `driver_version`: NVIDIA driver version;
- `cuda_version`: CUDA toolkit or runtime version;
- `architecture`: target architecture such as `sm_120a`;
- `operator_signature`: full operator-signature string, including shapes and
  data types;
- `selected_kernel`: selected registry kernel name;
- `provenance_kind`: project-native, imported, or other explicit provenance
  category;
- `upstream_revision`: source revision for imported kernels, or `null` for
  project-native kernels;
- `correctness_passed`: independent oracle pass/fail;
- `max_abs_error`: maximum absolute error;
- `max_rel_error`: maximum relative error;
- `cosine_similarity`: output cosine similarity;
- `nonfinite_mismatches`: count of non-finite mismatches;
- `warmup_count`: warmup launches before measurement;
- `measured_iterations`: measured timing iterations;
- `median_us`: median latency in microseconds;
- `p95_us`: p95 latency in microseconds;
- `min_us`: minimum measured latency in microseconds;
- `max_us`: maximum measured latency in microseconds;
- `workspace_bytes`: workspace bytes available to the selected launch;
- `launch_count`: launches per measured iteration;
- `graph_mode`: graph mode used for the measurement;
- `performance_publishable`: derived publication gate.

Required identity strings must be nonempty before serialization. Error metrics
must be finite and nonnegative, cosine similarity must be finite and within
`[-1, 1]`, timing fields must be finite, and measured timing records must have
positive `min_us`, `median_us`, `p95_us`, `max_us`, and `launch_count`. Latency
ordering must satisfy `min_us <= median_us <= p95_us <= max_us`.

Invalid metrics, timings, identity fields, or count combinations make the record
invalid rather than publishable evidence.

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
