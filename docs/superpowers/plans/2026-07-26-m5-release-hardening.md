# M5 Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the independently accepted BF16 and Q4_K_M Qwen3.5-9B paths into a reproducible, provenance-complete RTX 50 v0.1 release.

**Architecture:** Keep correctness, performance, provenance, and packaging as separate evidence producers, then combine them in one fail-closed release manifest. Runtime diagnostics are emitted from C++ through the existing coarse C ABI and Rust CLI boundary. Shell gates validate immutable JSON evidence and never infer an optimized path from throughput alone.

**Tech Stack:** C++20, CUDA 13, RAFT 26.06, RMM 26.06, Rust 2024, CMake/Ninja, Cargo, Bash, jq, CTest, Docker/Podman-compatible OCI image.

## Global Constraints

- v0.1 supports RTX 50-series `sm_120a` only.
- BF16 and Q4_K_M must each pass PP128, PP512, and TG128 after PP512 at `>=1.0x` llama.cpp, with at least one metric per format at `>=1.1x`.
- The same pinned model bytes, prompt token IDs, generated-token count, cache policy, and llama.cpp revision are required within each comparison arm.
- Exact greedy-token parity and layered numerical tests must pass before performance evidence is publishable.
- A benchmark is invalid when another compute process appears before, during, or immediately after a measured arm.
- CUDA Graph mode, attention implementation, matrix implementation, cache dtype/layout, quantized repack version, fallback reason, and peak RMM allocation are explicit evidence fields.
- Imported bw24 source must retain its exact upstream commit, source path, license, copyright, and local modification record.
- No release script creates a tag, GitHub release, or remote side effect.
- No new runtime dependency or second execution runtime is introduced.

---

## Planned File Structure

```text
cpp/
├── benchmarks/
│   ├── benchmark_record.hpp
│   └── benchmark_record.cpp
├── execution/
│   ├── qwen35_executor.hpp
│   └── qwen35_executor.cu
└── tests/
    ├── benchmark_record_test.cpp
    └── qwen35_executor_test.cu
rust/
├── raftinfer-runtime/src/lib.rs
└── raftinfer-cli/
    ├── src/main.rs
    └── tests/cli.rs
scripts/
├── gpu-preflight.sh
├── gpu-snapshot.sh
├── qwen35-benchmark.sh
├── v01-gate.sh
└── v01-release-check.sh
tests/
├── fixtures/release/
│   ├── bf16-pass.jsonl
│   ├── q4-k-m-pass.jsonl
│   ├── parity-pass.jsonl
│   └── provenance-pass.json
├── gpu-preflight-test.sh
├── gpu-snapshot-test.sh
├── v01-gate-script-test.sh
└── v01-release-check-test.sh
docs/
├── m5-verification.md
├── release/v0.1.md
└── provenance/
    ├── dependencies.md
    └── bw24-qwen35-kernels.md
THIRD_PARTY_LICENSES/
└── bw24-MIT.txt
```

### Task 1: Version the complete runtime and benchmark diagnostics

**Files:**
- Modify: `cpp/benchmarks/benchmark_record.hpp`
- Modify: `cpp/benchmarks/benchmark_record.cpp`
- Modify: `cpp/tests/benchmark_record_test.cpp`
- Modify: `cpp/execution/qwen35_executor.hpp`
- Modify: `cpp/execution/qwen35_executor.cu`
- Modify: `cpp/tests/qwen35_executor_test.cu`

**Interfaces:**
- Extends the runtime diagnostics with stable string values:

```cpp
struct Qwen35ExecutionDiagnostics {
  Qwen35AttentionImplementation attention;
  Qwen35KvCacheDType kv_cache_dtype;
  Qwen35KvCacheLayout kv_cache_layout;
  bool decode_graph_captured{};
  bool decode_graph_replayed{};
  std::size_t attention_workspace_bytes{};
  std::string attention_kernel;
  std::string matrix_kernel_digest;
  std::string weight_format;
  std::string cache_dtype;
  std::string cache_layout;
  std::string graph_mode;
  std::string graph_fallback_reason;
  std::uint32_t quant_repack_version{};
  std::uint64_t peak_rmm_bytes{};
  bool graph_capture_attempted{};
  bool optimized_path_complete{};
};
```

- Extends `BenchmarkRecord`:

```cpp
inline constexpr std::uint32_t benchmark_schema_version = 2;

struct BenchmarkRecord {
  std::uint32_t schema_version{benchmark_schema_version};
  std::string binary_sha256;
  std::string model_sha256;
  std::string reference_revision;
  std::string prompt_sha256;
  std::string weight_format;
  std::string cache_dtype;
  std::string cache_layout;
  std::string attention_kernel;
  std::string matrix_kernel_digest;
  std::string fallback_reason;
  std::uint32_t quant_repack_version{};
  std::uint64_t peak_rmm_bytes{};
  double coefficient_of_variation{};
  bool optimized_path_complete{};
};

enum class Qwen35BenchmarkWorkload : std::uint8_t {
  pp128,
  pp512,
  tg128_after_pp512,
};

bool qwen35_release_publishable(
    const BenchmarkRecord& record, Qwen35BenchmarkWorkload workload);
```

- [ ] **Step 1: Add failing schema-validation tests**

Require schema version 2, 64-character lowercase SHA-256 fields, nonempty
reference revision, recognized `BF16`/`Q4_K_M` weight formats, recognized cache
dtype/layout, finite nonnegative coefficient of variation, and a nonempty
fallback reason whenever `optimized_path_complete` is false.

Add positive fixtures for a project-native BF16 path and an imported bw24
quantized path. Add negative fixtures for graph mode reported as replay without
`graph_replayed`, a Q4_K_M record with repack version omitted, and a publishable
record with an undisclosed fallback.

- [ ] **Step 2: Run the focused host tests and verify RED**

```bash
cmake --build build/host --target raftinfer_benchmark_record_test
ctest --test-dir build/host \
  -R '^raftinfer_benchmark_record_test$' --output-on-failure
```

Expected: compilation fails because schema-version-2 fields are absent.

- [ ] **Step 3: Implement schema version 2 and executor population**

Serialize every field in deterministic key order. Preserve the existing
operator-level `performance_publishable()` behavior and make the new
`qwen35_release_publishable()` require:

```cpp
return correctness_passed &&
       optimized_path_complete &&
       fallback_reason.empty() &&
       warmup_count >= 3 &&
       measured_iterations >= 7 &&
       coefficient_of_variation <= 0.03 &&
       median_us > 0.0 &&
       (workload != Qwen35BenchmarkWorkload::tg128_after_pp512 ||
        graph_mode == "replay");
```

Populate executor diagnostics at plan construction and graph capture. Kernel
names and the matrix digest are immutable after construction; graph replay
state and peak RMM bytes are session observations.

- [ ] **Step 4: Run host and target diagnostics tests**

```bash
scripts/local-check.sh
ctest --test-dir <build-root>/m5/cuda \
  -R 'raftinfer_(benchmark_record|qwen35_executor)_test' --output-on-failure
```

Expected: all schema tests pass, executor records the selected paths, and no
diagnostic collection allocates during decode.

- [ ] **Step 5: Commit**

```bash
git add cpp/benchmarks/benchmark_record.hpp \
  cpp/benchmarks/benchmark_record.cpp cpp/tests/benchmark_record_test.cpp \
  cpp/execution/qwen35_executor.hpp cpp/execution/qwen35_executor.cu \
  cpp/tests/qwen35_executor_test.cu
git commit -m "feat: version Qwen3.5 release diagnostics"
```

### Task 2: Expose schema-version-2 evidence through the coarse Rust API

**Files:**
- Modify: `cpp/include/raftinfer/c_api.h`
- Modify: `cpp/src/c_api.cpp`
- Modify: `rust/raftinfer-sys/src/lib.rs`
- Modify: `rust/raftinfer-runtime/src/lib.rs`
- Modify: `rust/raftinfer-runtime/tests/engine.rs`
- Modify: `rust/raftinfer-cli/src/main.rs`
- Modify: `rust/raftinfer-cli/tests/cli.rs`

**Interfaces:**
- Adds one request-level diagnostics read:

```c
typedef struct RaftInferQwen35DiagnosticsV1 {
  size_t struct_size;
  uint32_t schema_version;
  const char* weight_format;
  const char* cache_dtype;
  const char* cache_layout;
  const char* attention_kernel;
  const char* matrix_kernel_digest;
  const char* graph_mode;
  const char* fallback_reason;
  uint32_t quant_repack_version;
  uint64_t peak_rmm_bytes;
  uint8_t optimized_path_complete;
} RaftInferQwen35DiagnosticsV1;

RaftInferStatus raftinfer_session_qwen35_diagnostics(
    const RaftInferSessionHandle* session, RaftInferQwen35DiagnosticsV1* output);
```

- Rust copies the borrowed C strings into:

```rust
pub struct Qwen35Diagnostics {
    pub schema_version: u32,
    pub weight_format: String,
    pub cache_dtype: String,
    pub cache_layout: String,
    pub attention_kernel: String,
    pub matrix_kernel_digest: String,
    pub graph_mode: String,
    pub fallback_reason: String,
    pub quant_repack_version: u32,
    pub peak_rmm_bytes: u64,
    pub optimized_path_complete: bool,
}
```

- [ ] **Step 1: Add failing ABI and CLI tests**

Require null checking, `struct_size` compatibility, UTF-8 copying, lifetime
independence after the next decode call, and JSON output containing every
diagnostic field. Verify one `benchmark` CLI command still performs one
request-level FFI call per prefill/decode step rather than one call per operator.

- [ ] **Step 2: Run and verify RED**

```bash
cargo test -p raftinfer-runtime -p raftinfer-cli
```

Expected: diagnostics types and symbol are absent.

- [ ] **Step 3: Implement the read-only ABI**

Keep string storage owned by the C++ session and valid until the next session
mutation; copy immediately in Rust. Return `RAFTINFER_STATUS_INVALID_ARGUMENT` for a
null output or undersized structure. Do not expose raw device addresses or add
operator-level entry points.

- [ ] **Step 4: Run native and Rust suites**

```bash
scripts/local-check.sh
cargo test --workspace --locked
```

Expected: ABI source checks, runtime tests, and CLI JSON snapshots pass.

- [ ] **Step 5: Commit**

```bash
git add cpp/include/raftinfer/c_api.h cpp/src/c_api.cpp \
  rust/raftinfer-sys/src/lib.rs rust/raftinfer-runtime/src/lib.rs \
  rust/raftinfer-runtime/tests/engine.rs rust/raftinfer-cli/src/main.rs \
  rust/raftinfer-cli/tests/cli.rs
git commit -m "feat: expose Qwen3.5 execution diagnostics"
```

### Task 3: Make benchmark evidence fail closed around GPU contention

**Files:**
- Create: `scripts/gpu-snapshot.sh`
- Create: `tests/gpu-snapshot-test.sh`
- Modify: `scripts/gpu-preflight.sh`
- Modify: `tests/gpu-preflight-test.sh`
- Modify: `scripts/qwen35-benchmark.sh`
- Modify: `tests/benchmark-script-test.sh`

**Interfaces:**
- `gpu-snapshot.sh` emits one JSON object:

```json
{
  "schema_version": 1,
  "gpu_index": 0,
  "uuid": "GPU-00000000-0000-0000-0000-000000000000",
  "name": "NVIDIA GeForce RTX 5090",
  "compute_capability": "12.0",
  "driver_version": "570.00",
  "free_mib": 16384,
  "utilization_percent": 0,
  "temperature_c": 35,
  "compute_apps": []
}
```

- [ ] **Step 1: Add mocked `nvidia-smi` fixtures**

Cover one idle RTX 5090, wrong compute capability, malformed CSV, one unrelated
compute process, the benchmark's own PID, utilization above the threshold, and
a process appearing only in the post-arm snapshot.

- [ ] **Step 2: Run and verify RED**

```bash
tests/gpu-snapshot-test.sh
tests/gpu-preflight-test.sh
tests/benchmark-script-test.sh
```

Expected: snapshot script is absent and post-arm contention is not detected.

- [ ] **Step 3: Implement snapshot and three-point arm validation**

Take snapshots immediately before an arm, after warmups, and after measurements.
The measured arm is valid only when all snapshots have the same GPU UUID,
compute capability `12.0`, no foreign compute application, and utilization
within the configured idle threshold before timing. Preserve `gpu-preflight.sh`
as the human-readable refusal wrapper over the JSON snapshot.

- [ ] **Step 4: Interleave RAFTINFER and llama.cpp measured arms**

For each workload execute:

```text
RAFTINFER warmup -> RAFTINFER seven measurements -> idle snapshot
llama warmup -> llama seven measurements -> idle snapshot
llama warmup -> llama seven measurements -> idle snapshot
RAFTINFER warmup -> RAFTINFER seven measurements -> idle snapshot
```

Record both arm medians and use the median of the valid samples for the ratio.
Abort the workload instead of averaging any contaminated arm.

- [ ] **Step 5: Run script tests**

Expected: all contention fixtures fail closed, the benchmark accepts its own
known process IDs only, and the idle fixture emits schema-version-2 evidence.

- [ ] **Step 6: Commit**

```bash
git add scripts/gpu-snapshot.sh scripts/gpu-preflight.sh \
  scripts/qwen35-benchmark.sh tests/gpu-snapshot-test.sh \
  tests/gpu-preflight-test.sh tests/benchmark-script-test.sh
git commit -m "test: reject contaminated Qwen3.5 benchmarks"
```

### Task 4: Consolidate BF16 and Q4_K_M into one release gate

**Files:**
- Create: `scripts/v01-gate.sh`
- Create: `tests/v01-gate-script-test.sh`
- Create: `tests/fixtures/release/bf16-pass.jsonl`
- Create: `tests/fixtures/release/q4-k-m-pass.jsonl`
- Create: `tests/fixtures/release/parity-pass.jsonl`
- Create: `tests/fixtures/release/provenance-pass.json`
- Modify: `scripts/local-check.sh`

**Interfaces:**
- Invocation:

```bash
scripts/v01-gate.sh \
  --bf16 build/evidence/qwen35-bf16.jsonl \
  --q4-k-m build/evidence/qwen35-q4-k-m.jsonl \
  --bf16-parity build/evidence/qwen35-bf16-parity.jsonl \
  --q4-parity build/evidence/qwen35-q4-parity.jsonl \
  --provenance build/evidence/provenance.json \
  --output build/evidence/v0.1-gate.json
```

- Output:

```json
{
  "schema_version": 1,
  "release": "v0.1",
  "supported_architectures": ["sm_120a"],
  "bf16": {"passed": true},
  "q4_k_m": {"passed": true},
  "provenance": {"passed": true},
  "release_passed": true
}
```

- [ ] **Step 1: Add pass and independent-failure fixtures**

Cover every BF16/Q4 performance threshold, missing independent `1.1x`, exact
parity failure, different model hash within an arm, different prompt hash,
different llama.cpp revision, graph fallback, materialized attention, unknown
matrix kernel, coefficient of variation above three percent, fewer than three
warmups, fewer than seven measurements, wrong architecture, and missing bw24
license metadata.

- [ ] **Step 2: Run and verify RED**

```bash
tests/v01-gate-script-test.sh
```

Expected: `scripts/v01-gate.sh` is absent.

- [ ] **Step 3: Implement compositional gate validation**

Invoke the M3 and M4 gate scripts for their format-specific rules, then perform
cross-evidence checks. Require exactly one PP128 and one PP512 record per
format, TG128 attached to the PP512 record, complete parity corpus prompt IDs,
and provenance for every non-project-native selected kernel.

Write the output via a temporary file and atomic rename only after every check
passes. On failure, emit the failing JSON path to stderr and do not leave a
passing manifest.

- [ ] **Step 4: Add the release gate tests to local checks**

Expected: all fixture cases have stable exit codes and `scripts/local-check.sh`
runs the release-gate test without target hardware.

- [ ] **Step 5: Commit**

```bash
git add scripts/v01-gate.sh scripts/local-check.sh \
  tests/v01-gate-script-test.sh tests/fixtures/release
git commit -m "test: combine BF16 and Q4_K_M release gates"
```

### Task 5: Close imported-source and dependency provenance

**Files:**
- Modify: `docs/provenance/dependencies.md`
- Modify: `docs/provenance/bw24-qwen35-kernels.md`
- Modify: `NOTICE`
- Create when bw24 source is imported: `THIRD_PARTY_LICENSES/bw24-MIT.txt`
- Create when bw24 source is imported: `cpp/kernels/imported/bw24/README.md`
- Create when bw24 source is imported: `cpp/kernels/imported/bw24/LICENSE`
- Create: `scripts/verify-provenance.sh`
- Create: `tests/verify-provenance-test.sh`

**Interfaces:**
- `verify-provenance.sh` scans tracked source under
  `cpp/kernels/imported/bw24/` and emits:

```json
{
  "schema_version": 1,
  "upstream": "https://github.com/avifenesh/bw24",
  "upstream_revision": "0000000000000000000000000000000000000000",
  "license": "MIT",
  "imported_files": [],
  "local_modifications_documented": true,
  "passed": true
}
```

- [ ] **Step 1: Add failing provenance trees**

Test no imported source, a valid imported source, missing license text, missing
upstream path, missing commit, unlisted imported file, stale listed file, and
an undocumented local modification. The no-import case passes with an empty
file list and no bw24 notice requirement.

- [ ] **Step 2: Run and verify RED**

```bash
tests/verify-provenance-test.sh
```

Expected: verification script is absent.

- [ ] **Step 3: Record exact dependencies and imports**

Pin RAFT, RMM, CUDA toolkit, compiler, Rust toolchain, container base digest,
llama.cpp reference commit/build flags, Qwen3.5 model revision, Q4_K_M converter
revision, and each imported bw24 source. Copy the upstream MIT text verbatim
when source is imported and add a concise NOTICE entry.

- [ ] **Step 4: Implement tracked-file verification**

Use `git ls-files` as the source inventory. Require every imported `.cu`,
`.cuh`, and header to have an exact upstream path or explicit
`project-adapter` classification, and require every local semantic change to be
listed. Do not accept branch names or `HEAD` as revisions.

- [ ] **Step 5: Run provenance and license checks**

Expected: provenance passes for the actual tree and fails every malformed
fixture.

- [ ] **Step 6: Commit**

```bash
git add docs/provenance/dependencies.md \
  docs/provenance/bw24-qwen35-kernels.md NOTICE \
  scripts/verify-provenance.sh tests/verify-provenance-test.sh
if test -d THIRD_PARTY_LICENSES; then
  git add THIRD_PARTY_LICENSES
fi
if test -d cpp/kernels/imported/bw24; then
  git add cpp/kernels/imported/bw24
fi
git commit -m "docs: close Qwen3.5 kernel provenance"
```

### Task 6: Document and test the reproducible v0.1 build

**Files:**
- Modify: `containers/Dockerfile.dev`
- Modify: `tests/dockerfile-dev-test.sh`
- Modify: `README.md`
- Create: `docs/release/v0.1.md`
- Create: `scripts/v01-release-check.sh`
- Create: `tests/v01-release-check-test.sh`

**Interfaces:**
- `v01-release-check.sh` accepts:

```bash
scripts/v01-release-check.sh \
  --build-dir build/release \
  --gate-manifest build/evidence/v0.1-gate.json \
  --provenance-manifest build/evidence/provenance.json \
  --evidence-commit "$(jq -r .project_commit build/evidence/v0.1-gate.json)"
```

- [ ] **Step 1: Add release-check failure fixtures**

Cover dirty tree, wrong branch commit in the evidence, missing container digest,
wrong CUDA/RAFT/RMM version, absent `sm_120a` code object, missing Rust binary,
missing shared library, failing local test, failing CUDA test manifest, failed
release gate, and README claiming RTX 40 or unsupported model features.

- [ ] **Step 2: Run and verify RED**

```bash
tests/v01-release-check-test.sh
tests/dockerfile-dev-test.sh
```

Expected: release-check script is absent.

- [ ] **Step 3: Pin build inputs and write exact instructions**

Pin the OCI base by digest while retaining the readable tag in a comment.
Document:

```bash
cmake -S . -B build/release -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
cargo build --workspace --release --locked
ctest --test-dir build/release --output-on-failure
```

State the supported matrix exactly: Qwen3.5-9B text, BF16 and Q4_K_M, batch 1,
greedy decode, RTX 50 `sm_120a`. List multimodal, RTX 40, batching, paged KV,
MTP/speculative decode, multi-GPU, and mandatory NVFP4 as unsupported.

- [ ] **Step 4: Implement non-mutating release validation**

Verify tool versions, code object architecture with `cuobjdump --list-elf`,
tracked-tree cleanliness, binary/library presence, local and target test
summaries, gate manifest, provenance manifest, README support claims, and
presence of all required verification documents. The evidence commit must equal
the built binary commit. The current clean commit may be a descendant only when
`git diff --name-only <evidence-commit>..HEAD` contains exclusively
`docs/m5-verification.md` and `docs/release/v0.1.md`. The script does not tag or
publish.

- [ ] **Step 5: Run local release fixtures**

Expected: every malformed fixture fails with a named check and the valid
fixture passes.

- [ ] **Step 6: Commit**

```bash
git add containers/Dockerfile.dev tests/dockerfile-dev-test.sh README.md \
  docs/release/v0.1.md scripts/v01-release-check.sh \
  tests/v01-release-check-test.sh
git commit -m "docs: add reproducible v0.1 release workflow"
```

### Task 7: Produce final target evidence and the release checklist

**Files:**
- Create: `docs/m5-verification.md`
- Modify: `docs/release/v0.1.md`

**Interfaces:**
- The final verification document records:
  - project commit and clean-tree status;
  - target GPU UUID/name/compute capability and driver;
  - container, CUDA, RAFT, RMM, compiler, and Rust versions;
  - BF16 and Q4_K_M model hashes;
  - llama.cpp revision and build flags;
  - operator, executor, state-transition, tokenizer, and exact-token results;
  - PP128, PP512, and TG128@PP512 medians, dispersion, and ratios per format;
  - attention, matrix, graph, cache, repack, fallback, and peak-RMM diagnostics;
  - imported-source provenance result;
  - known limitations and deferred features.

- [ ] **Step 1: Run the complete local suite**

```bash
scripts/local-check.sh
cargo test --workspace --locked
git diff --check
```

Expected: all host, Rust, source-contract, and shell-fixture tests pass.

- [ ] **Step 2: Sync the exact commit to the target**

```bash
scripts/sync-target.sh
```

Expected: target checkout reports the same project commit and has no untracked
source override.

- [ ] **Step 3: Build and run the target CUDA suite**

```bash
cmake --build <build-root>/m5/cuda
ctest --test-dir <build-root>/m5/cuda --output-on-failure
```

Expected: all CUDA tests pass on compute capability 12.0.

- [ ] **Step 4: Generate fresh correctness and benchmark evidence**

```bash
scripts/qwen35-parity.sh
scripts/qwen35-q4-parity.sh
env \
  RAFTINFER_WEIGHT_FORMAT=BF16 \
  RAFTINFER_MODEL=<artifact-root>/Qwen3.5-9B-GGUF/Qwen3.5-9B-c202236-bf16.gguf \
  LLAMA_MODEL=<artifact-root>/Qwen3.5-9B-GGUF/Qwen3.5-9B-c202236-bf16.gguf \
  PARITY_REPORT=build/evidence/qwen35-bf16-parity.jsonl \
  BENCHMARK_OUTPUT=build/evidence/qwen35-bf16.jsonl \
  scripts/qwen35-benchmark.sh
env \
  RAFTINFER_WEIGHT_FORMAT=Q4_K_M \
  RAFTINFER_MODEL=<artifact-root>/Qwen3.5-9B-GGUF/Qwen3.5-9B-c202236-q4_k_m.gguf \
  LLAMA_MODEL=<artifact-root>/Qwen3.5-9B-GGUF/Qwen3.5-9B-c202236-q4_k_m.gguf \
  PARITY_REPORT=build/evidence/qwen35-q4-parity.jsonl \
  BENCHMARK_OUTPUT=build/evidence/qwen35-q4-k-m.jsonl \
  scripts/qwen35-benchmark.sh
scripts/qwen35-bf16-gate.sh build/evidence/qwen35-bf16.jsonl
scripts/qwen35-q4-gate.sh build/evidence/qwen35-q4-k-m.jsonl
scripts/verify-provenance.sh > build/evidence/provenance.json
scripts/v01-gate.sh \
  --bf16 build/evidence/qwen35-bf16.jsonl \
  --q4-k-m build/evidence/qwen35-q4-k-m.jsonl \
  --bf16-parity build/evidence/qwen35-bf16-parity.jsonl \
  --q4-parity build/evidence/qwen35-q4-parity.jsonl \
  --provenance build/evidence/provenance.json \
  --output build/evidence/v0.1-gate.json
```

Expected: both mandatory formats independently pass correctness, exact-token,
and performance gates with no foreign GPU process and no undisclosed fallback.

- [ ] **Step 5: Complete the non-mutating release check**

```bash
scripts/v01-release-check.sh \
  --build-dir <build-root>/m5/cuda \
  --gate-manifest build/evidence/v0.1-gate.json \
  --provenance-manifest build/evidence/provenance.json \
  --evidence-commit "$(jq -r .project_commit build/evidence/v0.1-gate.json)"
```

Expected: all release checks pass. If any gate fails, record the measured result
and leave the corresponding release checklist item unchecked.

- [ ] **Step 6: Write the evidence documents**

Copy only summarized, immutable evidence into `docs/m5-verification.md`; do not
invent passing values. Update `docs/release/v0.1.md` so every checked item has a
documented command and artifact hash.

- [ ] **Step 7: Commit the evidence documents**

```bash
git add docs/m5-verification.md docs/release/v0.1.md
git commit -m "docs: record v0.1 release evidence"
```

- [ ] **Step 8: Re-run document and tree checks**

```bash
scripts/local-check.sh
scripts/v01-release-check.sh \
  --build-dir <build-root>/m5/cuda \
  --gate-manifest build/evidence/v0.1-gate.json \
  --provenance-manifest build/evidence/provenance.json \
  --evidence-commit "$(jq -r .project_commit build/evidence/v0.1-gate.json)"
git diff --check
```

Expected: checks pass against the final documented tree.
