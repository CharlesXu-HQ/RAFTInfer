# M1 Registry and Correctness Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver host-tested tensor/operator contracts, deterministic kernel dispatch, preallocated workspace layout, independent CPU reference operators, correctness metrics, and benchmark JSONL for later Qwen3 Dense CUDA kernels.

**Architecture:** Public tensor metadata remains an ABI-safe C contract while registry, reference, and benchmark APIs remain private C++20. CUDA-only execution context and RMM workspace ownership sit behind host-testable layout arithmetic. Every optimized kernel added later must resolve through the registry and be compared with an independent CPU reference before its performance record is valid.

**Tech Stack:** C++20, CMake 3.30.4+, CTest, CUDA 13.x, RAFT 26.06, RMM 26.06, Rust 1.96.0, Cargo, RTX 50 `sm_120a`.

## Global Constraints

- v0.1 targets RTX 50-series consumer Blackwell only; compile CUDA code for `sm_120a`.
- RAFT/RMM own common GPU resource and memory infrastructure; custom CUDA owns hot kernels.
- C++ owns every GPU pointer, stream, event, allocation, graph, and workspace.
- Rust uses coarse opaque-handle FFI and does not call individual operators.
- CPU reference code must not share unpacking, loop structure, or numerical shortcuts with future CUDA kernels.
- Correctness and algorithmic invariants gate performance publication.
- No `bw24` source is imported in M1.
- Already optimized `bw24` kernels may later be reused directly only after license, provenance, correctness, algorithmic, and RTX 50 performance validation.
- Never stop or alter unrelated GPU processes on `<validation-root>`; CUDA validation runs only after the existing fail-closed preflight succeeds.

## Planned file structure

```text
cpp/
├── include/raftinfer/
│   ├── status.h
│   └── tensor.h
├── operators/
│   ├── tensor_validation.cpp
│   └── tensor_validation.hpp
├── execution/
│   ├── execution_context.hpp
│   ├── workspace_arena.cu
│   ├── workspace_arena.hpp
│   └── workspace_layout.hpp
├── registry/
│   ├── operator_registry.cpp
│   └── operator_registry.hpp
├── reference/
│   ├── bf16.hpp
│   ├── correctness.cpp
│   ├── correctness.hpp
│   ├── operators.cpp
│   └── operators.hpp
├── benchmarks/
│   ├── benchmark_record.cpp
│   └── benchmark_record.hpp
└── tests/
    ├── benchmark_record_test.cpp
    ├── correctness_test.cpp
    ├── operator_registry_test.cpp
    ├── reference_operators_test.cpp
    ├── tensor_validation_test.cpp
    └── workspace_layout_test.cpp
```

---

### Task 1: ABI-safe tensor descriptors and validation

**Files:**
- Modify: `cpp/include/raftinfer/status.h`
- Create: `cpp/include/raftinfer/tensor.h`
- Create: `cpp/operators/tensor_validation.hpp`
- Create: `cpp/operators/tensor_validation.cpp`
- Create: `cpp/tests/tensor_validation_test.cpp`
- Modify: `cpp/CMakeLists.txt`
- Modify: `rust/raftinfer-sys/src/lib.rs`

**Interfaces:**
- Produces: `RaftInferDataType`, `RaftInferQuantFormat`, `RaftInferMemoryType`, `RaftInferTensorDesc`, and `raftinfer::validate_tensor_desc(const RaftInferTensorDesc&)`.
- Consumes: `RaftInferStatusCode`.

- [ ] **Step 1: Write the failing descriptor test**

Create `cpp/tests/tensor_validation_test.cpp` with cases that construct a
contiguous rank-two FP32 descriptor, reject rank five, reject zero dimensions,
reject an undersized byte range, and accept an opaque Q4_K descriptor with
scale metadata:

```cpp
#include "../operators/tensor_validation.hpp"

#include <cassert>
#include <stdexcept>

int main() {
  float values[6]{};
  RaftInferTensorDesc valid{
      values, nullptr, nullptr, 24, {2, 3, 0, 0}, {12, 4, 0, 0},
      2, RAFTINFER_DTYPE_F32, RAFTINFER_QUANT_NONE, RAFTINFER_MEMORY_HOST};
  raftinfer::validate_tensor_desc(valid);

  auto invalid = valid;
  invalid.rank = 5;
  try { raftinfer::validate_tensor_desc(invalid); assert(false); }
  catch (const std::invalid_argument&) {}

  invalid = valid;
  invalid.shape[1] = 0;
  try { raftinfer::validate_tensor_desc(invalid); assert(false); }
  catch (const std::invalid_argument&) {}

  invalid = valid;
  invalid.byte_size = 20;
  try { raftinfer::validate_tensor_desc(invalid); assert(false); }
  catch (const std::invalid_argument&) {}

  unsigned char packed[32]{};
  float scales[2]{};
  RaftInferTensorDesc quant{
      packed, scales, nullptr, sizeof(packed), {2, 32, 0, 0}, {16, 0, 0, 0},
      2, RAFTINFER_DTYPE_Q4_K, RAFTINFER_QUANT_Q4_K, RAFTINFER_MEMORY_HOST};
  raftinfer::validate_tensor_desc(quant);
}
```

- [ ] **Step 2: Register and run the test to verify it fails**

Add a `raftinfer_tensor_validation_test` executable and CTest entry in
`cpp/CMakeLists.txt`.

Run:

```bash
cmake -S . -B build/host -G Ninja -DRAFTINFER_ENABLE_CUDA=OFF
cmake --build build/host --target raftinfer_tensor_validation_test
```

Expected: compilation fails because `tensor_validation.hpp` is absent.

- [ ] **Step 3: Implement public metadata and validation**

Add status values with stable integers:

```c
RAFTINFER_STATUS_UNSUPPORTED = 5,
RAFTINFER_STATUS_RESOURCE_EXHAUSTED = 6
```

Define the enums and descriptor in `cpp/include/raftinfer/tensor.h`. Use explicit
integer values, fixed arrays of length four, and this field order:

```c
typedef struct RaftInferTensorDesc {
  void* data;
  const void* scales;
  const void* zero_points;
  size_t byte_size;
  int64_t shape[4];
  int64_t strides[4];
  uint32_t rank;
  RaftInferDataType dtype;
  RaftInferQuantFormat quant;
  RaftInferMemoryType memory;
} RaftInferTensorDesc;
```

Implement checked byte-bound validation for unquantized tensors. For quantized
tensors validate non-null data, non-null scales, positive dimensions, known
enums, and non-zero bytes without interpreting packed blocks.

Mirror the new status integers in `rust/raftinfer-sys/src/lib.rs`.

- [ ] **Step 4: Run tensor and ABI tests**

Run:

```bash
cmake --build build/host --target raftinfer_tensor_validation_test raftinfer_c_api_test
ctest --test-dir build/host -R 'raftinfer_(tensor_validation|c_api)_test' --output-on-failure
cargo test -p raftinfer-runtime
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add cpp/include/raftinfer/status.h cpp/include/raftinfer/tensor.h cpp/operators cpp/tests/tensor_validation_test.cpp cpp/CMakeLists.txt rust/raftinfer-sys/src/lib.rs
git commit -m "feat: add tensor metadata contract"
```

---

### Task 2: Host-testable workspace layout and CUDA-owned arena

**Files:**
- Create: `cpp/execution/workspace_layout.hpp`
- Create: `cpp/execution/workspace_arena.hpp`
- Create: `cpp/execution/workspace_arena.cu`
- Create: `cpp/execution/execution_context.hpp`
- Create: `cpp/tests/workspace_layout_test.cpp`
- Modify: `cpp/foundation/device_context.hpp`
- Modify: `cpp/foundation/device_context.cu`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Produces: `WorkspaceLayout::allocate(size_t,size_t)`, `reset()`, `used()`,
  `capacity()`, and CUDA-only `WorkspaceArena`.
- Consumes: the engine-owned RMM pool and RAFT resources.

- [ ] **Step 1: Write the failing layout test**

```cpp
#include "../execution/workspace_layout.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>

int main() {
  raftinfer::WorkspaceLayout layout{256};
  assert(layout.allocate(3, 1) == 0);
  assert(layout.allocate(16, 16) == 16);
  assert(layout.used() == 32);
  try { (void)layout.allocate(1, 3); assert(false); }
  catch (const std::invalid_argument&) {}
  try { (void)layout.allocate(240, 16); assert(false); }
  catch (const std::length_error&) {}
  layout.reset();
  assert(layout.used() == 0);
  assert(layout.allocate(256, 256) == 0);
}
```

- [ ] **Step 2: Run it and verify the missing-header failure**

Register `raftinfer_workspace_layout_test`, configure, and build that target.
Expected: compilation fails because `workspace_layout.hpp` is absent.

- [ ] **Step 3: Implement checked layout arithmetic**

Implement `WorkspaceLayout` as a small value type. Alignment must be a non-zero
power of two. Reject `offset + padding + bytes` overflow and capacity overflow.
Zero-byte allocations return the current aligned offset without advancing it.

- [ ] **Step 4: Implement the CUDA arena and execution context**

`WorkspaceArena` allocates one byte buffer from
`rmm::device_async_resource_ref` at construction and deallocates it on the same
stream at destruction. `allocate` returns `base + offset`. `reset` changes only
the host-side offset.

`ExecutionContext` stores non-owning references to `raft::device_resources`,
`rmm::device_async_resource_ref`, `cudaStream_t`, `WorkspaceArena`, device ID,
compute capability major/minor, and maximum shared memory per block.

Extend `DeviceContext::Resources` so destruction order remains RAFT resources,
workspace arena, RMM pool, CUDA upstream. Add a 1 MiB arena and a private
allocation/reset probe invoked by the existing smoke path.

- [ ] **Step 5: Run host and CUDA source checks**

Run:

```bash
cmake --build build/host --target raftinfer_workspace_layout_test
ctest --test-dir build/host -R 'raftinfer_(workspace_layout|device_context_source)_test' --output-on-failure
```

Expected: selected tests pass. The target GPU build is deferred to Task 7.

- [ ] **Step 6: Commit**

```bash
git add cpp/execution cpp/tests/workspace_layout_test.cpp cpp/foundation cpp/CMakeLists.txt
git commit -m "feat: add preallocated workspace arena"
```

---

### Task 3: Deterministic operator registry

**Files:**
- Create: `cpp/registry/operator_registry.hpp`
- Create: `cpp/registry/operator_registry.cpp`
- Create: `cpp/tests/operator_registry_test.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Produces: `OperatorSignature`, `KernelCapability`, `KernelRegistration`,
  `DispatchResult`, and `OperatorRegistry::resolve`.
- Consumes: tensor enums and workspace availability from Tasks 1 and 2.

- [ ] **Step 1: Write failing dispatch tests**

Register three metadata-only kernels: a high-priority graph-unsafe kernel, a
lower-priority graph-safe kernel, and a deterministic fallback. Assert:

```cpp
auto ordinary = registry.resolve(signature(false, true, 4096));
assert(ordinary.registration->name == "fast_ordinary");
auto graph = registry.resolve(signature(true, true, 4096));
assert(graph.registration->name == "graph_safe");
auto deterministic = registry.resolve(signature(false, true, 0));
assert(deterministic.registration->name == "no_workspace_fallback");
assert(&registry.resolve(signature(false, true, 4096)).registration.value().get()
       == &ordinary.registration.value().get());
```

Also assert duplicate names fail, unsupported signatures contain rejection
reasons, and nondeterministic kernels cannot satisfy deterministic requests.

- [ ] **Step 2: Run and verify the missing-header failure**

Register `raftinfer_operator_registry_test` and build it. Expected: compilation fails
because `operator_registry.hpp` is absent.

- [ ] **Step 3: Implement registry types and matching**

Use enum classes for operator kind and execution regime. Store tensor metadata
in comparable value types rather than pointer-bearing `RaftInferTensorDesc`.
`OperatorSignature` implements equality and a complete hash.

`KernelCapability::matches` checks operator, regime, architecture, dtype,
quantization, rank, alignment, shape bounds, graph safety, determinism, and
workspace bytes. `OperatorRegistry::resolve` filters, sorts by descending
priority then ascending name, stores the winning registration in a cache, and
throws `DispatchError` with per-kernel reasons when no match exists.

- [ ] **Step 4: Run dispatch tests twice**

```bash
cmake --build build/host --target raftinfer_operator_registry_test
ctest --test-dir build/host -R raftinfer_operator_registry_test --repeat until-fail:2 --output-on-failure
```

Expected: both deterministic runs pass.

- [ ] **Step 5: Commit**

```bash
git add cpp/registry cpp/tests/operator_registry_test.cpp cpp/CMakeLists.txt
git commit -m "feat: add deterministic operator registry"
```

---

### Task 4: Independent CPU reference operators

**Files:**
- Create: `cpp/reference/bf16.hpp`
- Create: `cpp/reference/operators.hpp`
- Create: `cpp/reference/operators.cpp`
- Create: `cpp/tests/reference_operators_test.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Produces: `rms_norm`, `rope`, `bf16_linear`, `softmax`, `argmax`,
  `embedding`, `add`, and `swiglu` under `raftinfer::reference`.
- Consumes: standard C++ spans and explicit dimension structs only.

- [ ] **Step 1: Write deterministic fixture tests**

Add exact fixtures for Add, embedding, and argmax tie-breaking. Add numerical
fixtures for RMSNorm, RoPE position zero and one, BF16 identity linear,
max-subtracted softmax, and SwiGLU. Assert invalid epsilon, odd rotary
dimension, out-of-range token ID, and mismatched spans throw
`std::invalid_argument`.

Use this seeded randomized loop for shape coverage:

```cpp
std::mt19937 rng{0xB124};
std::uniform_real_distribution<float> dist(-3.0F, 3.0F);
for (int trial = 0; trial < 32; ++trial) {
  std::vector<float> x(64);
  std::generate(x.begin(), x.end(), [&] { return dist(rng); });
  // Compare each simple reference with a separately written scalar expression.
}
```

- [ ] **Step 2: Build and verify failure**

Register `raftinfer_reference_operators_test` and build it. Expected: compilation
fails because `reference/operators.hpp` is absent.

- [ ] **Step 3: Implement BF16 conversion and references**

Implement round-to-nearest-even FP32-to-BF16 and a bit-exact BF16-to-FP32 using
`std::bit_cast`. Accumulate BF16 linear in FP32. Implement softmax with FP64
sum and first-index argmax tie behavior. All output spans must be fully
validated before writes begin.

- [ ] **Step 4: Run reference tests**

```bash
cmake --build build/host --target raftinfer_reference_operators_test
ctest --test-dir build/host -R raftinfer_reference_operators_test --output-on-failure
```

Expected: the fixture, randomized, and invalid-input cases pass.

- [ ] **Step 5: Commit**

```bash
git add cpp/reference cpp/tests/reference_operators_test.cpp cpp/CMakeLists.txt
git commit -m "feat: add CPU operator references"
```

---

### Task 5: Correctness metrics and operator-local gates

**Files:**
- Create: `cpp/reference/correctness.hpp`
- Create: `cpp/reference/correctness.cpp`
- Create: `cpp/tests/correctness_test.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Produces: `CorrectnessMetrics`, `Tolerance`, `compare`, and
  `passes_tolerance`.
- Consumes: candidate/reference FP32 spans.

- [ ] **Step 1: Write failing metric tests**

Test exact equality, bounded error, one non-finite mismatch, two all-zero
vectors, one zero vector, unequal span lengths, and exact-index equality.
Verify the relative-error floor prevents division by zero:

```cpp
auto metrics = raftinfer::reference::compare(
    std::array{1.0e-9F}, std::array{2.0e-9F}, 1.0e-6F);
assert(std::abs(metrics.max_relative_error - 0.001) < 1.0e-6);
```

- [ ] **Step 2: Build and verify failure**

Register `raftinfer_correctness_test` and build it. Expected: missing-header failure.

- [ ] **Step 3: Implement metrics**

`compare` reports maximum absolute error, maximum relative error, cosine
similarity, non-finite mismatch count, and element count. `passes_tolerance`
requires finite-compatible values and checks all enabled thresholds. Provide a
separate `indices_equal` for exact index spans.

- [ ] **Step 4: Run tests**

```bash
cmake --build build/host --target raftinfer_correctness_test
ctest --test-dir build/host -R raftinfer_correctness_test --output-on-failure
```

Expected: all metric semantics pass.

- [ ] **Step 5: Commit**

```bash
git add cpp/reference/correctness.hpp cpp/reference/correctness.cpp cpp/tests/correctness_test.cpp cpp/CMakeLists.txt
git commit -m "test: add operator correctness metrics"
```

---

### Task 6: Versioned benchmark JSONL

**Files:**
- Create: `cpp/benchmarks/benchmark_record.hpp`
- Create: `cpp/benchmarks/benchmark_record.cpp`
- Create: `cpp/tests/benchmark_record_test.cpp`
- Modify: `cpp/CMakeLists.txt`
- Create: `docs/m1-verification.md`
- Modify: `README.md`

**Interfaces:**
- Produces: `BenchmarkRecord::to_json_line()` and
  `BenchmarkRecord::performance_publishable()`.
- Consumes: registry identity and correctness metrics as plain record fields.

- [ ] **Step 1: Write failing JSONL tests**

Create a complete record containing a quoted kernel name and assert the output:

- ends in exactly one newline;
- contains `"schema_version":1`;
- escapes quotes and backslashes;
- contains no raw newline inside a JSON string;
- reports `performance_publishable()` only when correctness passed, measured
  iterations are non-zero, and median/p95 are finite and positive.

- [ ] **Step 2: Build and verify failure**

Register `raftinfer_benchmark_record_test` and build it. Expected: missing-header
failure.

- [ ] **Step 3: Implement the serializer**

Implement a deterministic field order and JSON escaping for control characters,
quotes, and backslashes. Reject non-finite metric or timing values before
serialization. Emit optional upstream revision as `null` for project-native
kernels.

- [ ] **Step 4: Update documentation**

Change README's current milestone to M1 and document that M1 supplies contracts
and CPU oracles but still does not load Qwen3 Dense. Create
`docs/m1-verification.md` with exact local and target commands, evidence fields,
and the distinction between correctness-valid and publishable performance.

- [ ] **Step 5: Run all local checks**

```bash
scripts/local-check.sh
git diff --check
```

Expected: all CTest and Cargo tests pass; formatting and diff checks pass.

- [ ] **Step 6: Commit**

```bash
git add cpp/benchmarks cpp/tests/benchmark_record_test.cpp cpp/CMakeLists.txt README.md docs/m1-verification.md
git commit -m "feat: add benchmark evidence records"
```

---

### Task 7: RTX 5090 CUDA validation and M1 evidence

**Files:**
- Modify: `scripts/gpu-smoke.sh`
- Modify: `docs/m1-verification.md`
- Modify: `docs/provenance/dependencies.md`

**Interfaces:**
- Produces: fresh M1 build/smoke evidence on `<validation-root>`.
- Consumes: all prior tasks and the existing fail-closed GPU preflight.

- [ ] **Step 1: Extend smoke validation**

Make the smoke build execute the workspace allocation/reset probe through the
existing `raftinfer_engine_run_smoke` path. Keep the golden JSON unchanged so M0 ABI
consumers remain compatible.

- [ ] **Step 2: Run remote preflight**

Use interactive SSH authentication, then run:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
```

Expected: no unrelated compute application and sufficient memory according to
`scripts/gpu-preflight.sh`. If not, stop without changing any process.

- [ ] **Step 3: Sync into a run-specific target directory and validate**

Run the existing non-destructive source staging method, build the pinned
container image if absent, and execute:

```bash
scripts/gpu-smoke.sh
```

Expected:

```json
{"device_id":0,"element_count":1024,"checksum":523776}
```

The output proves RAFT/RMM construction, the custom CUDA kernel, and the new
preallocated workspace allocation/reset probe completed.

- [ ] **Step 4: Record evidence and rerun local checks**

Record commit, image digest, GPU, driver, CUDA, RAFT/RMM versions, pre/post GPU
state, test output, and cleanup state in `docs/m1-verification.md`. Update
dependency provenance only if exact versions changed.

Run:

```bash
scripts/local-check.sh
git diff --check
git status --short
```

Expected: all checks pass and only intended evidence files differ.

- [ ] **Step 5: Commit**

```bash
git add scripts/gpu-smoke.sh docs/m1-verification.md docs/provenance/dependencies.md
git commit -m "docs: record M1 target verification"
```

---

### Task 8: Completion review

**Files:**
- Review all M1 files and commits.

**Interfaces:**
- Produces: a verified M1 milestone ready for the Qwen3 Dense baseline plan.
- Consumes: Tasks 1 through 7.

- [ ] **Step 1: Verify requirement coverage**

Map every success criterion in
`docs/superpowers/specs/2026-07-24-m1-registry-correctness-design.md` to a test
or target evidence entry. Treat any unmapped criterion as incomplete.

- [ ] **Step 2: Run final validation**

```bash
scripts/local-check.sh
git diff --check
git status --short --branch
```

Expected: all local checks pass and the worktree is clean.

- [ ] **Step 3: Review API and implementation**

Request a correctness/architecture review focused on ABI stability, complete
signature hashing, lifetime/destruction order, reference independence, JSON
validity, and whether any code permits performance publication before
correctness.

- [ ] **Step 4: Fix findings and revalidate**

For each accepted finding, add or strengthen a regression test before changing
implementation. Re-run the targeted test and the full local suite.

- [ ] **Step 5: Commit review fixes**

```bash
git add cpp rust scripts docs README.md
git commit -m "fix: address M1 completion review"
```

Skip this commit only when review reports no actionable findings and the
worktree is already clean.
