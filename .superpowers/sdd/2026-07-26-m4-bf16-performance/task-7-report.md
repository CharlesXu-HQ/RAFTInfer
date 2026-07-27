# Task 7 Report: Tune Cached Matrix Algorithms for Token Shape

## Summary

Implemented construction-time cuBLASLt candidate enumeration and median-timed
selection, then wired Qwen3.5 executor plan construction to tune only release
token buckets `1`, `128`, and `512`. Identical matmul shape/dtype/order plans
share the selected plan metadata through construction-time plan caching.

Status: DONE_WITH_CONCERNS because this host has no CUDA toolkit or RTX 50 GPU,
so CUDA target build/runtime RED/GREEN could not be executed locally.

## Files

- `cpp/execution/cublaslt_matmul.hpp`
- `cpp/execution/cublaslt_matmul.cu`
- `cpp/execution/qwen35_executor.cu`
- `cpp/tests/cublaslt_matmul_test.cu`
- `cpp/tests/qwen35_executor_test.cu`

## TDD RED

Tests added first:

- `cpp/tests/cublaslt_matmul_test.cu`
  - Calls `enumerate_cublaslt_candidates(config, 16)`.
  - Requires nonempty successful candidates, candidate count no greater than 16,
    nonnegative algorithm IDs, and workspace sizes within budget.
  - Runs the original first-heuristic plan and a tuned plan on the same real
    device inputs and compares the selected output to the first-heuristic
    output.
- `cpp/tests/qwen35_executor_test.cu`
  - Adds a test-only plan-selection diagnostic counter.
  - Requires construction-time selections to be recorded.
  - Resets counters before prefill, decode graph capture, and decode graph
    replay, then requires no selection/timing calls during those execution
    phases.

RED command attempted:

```bash
cmake --build build/host --target brt_cublaslt_matmul_test brt_qwen35_executor_test
```

Output:

```text
ninja: error: unknown target 'brt_cublaslt_matmul_test'
```

Reason: existing local configured tree is host-only (`BRT_ENABLE_CUDA=OFF`), so
the CUDA test targets are not generated.

CUDA configure command attempted:

```bash
cmake -S . -B build/cuda -G Ninja -DBRT_ENABLE_CUDA=ON -DBRT_BUILD_TESTS=ON
```

Output:

```text
Failed to find nvcc.
Compiler requires the CUDA toolkit. Please set the CUDAToolkit_ROOT variable.
```

Docker toolchain check:

```bash
docker images brt-dev:26.06-cuda13
```

Output:

```text
IMAGE   ID   DISK USAGE   CONTENT SIZE   EXTRA
```

The project CUDA development image is not present locally.

## GREEN Implementation

`cublaslt_matmul`:

- Added `CublasLtCandidate`.
- Added `enumerate_cublaslt_candidates(config, maximum = 16)`.
- Kept `CublasLtMatmulPlan::create` behavior as first successful heuristic.
- Added `CublasLtMatmulPlan::select_fastest(...)`.
- `select_fastest`:
  - Enumerates up to 16 candidates.
  - Rejects failed or over-budget candidates.
  - Validates caller-owned buffers and workspace.
  - Uses CUDA events for warmups and measurements.
  - Selects the lowest median timing.
  - Stores the selected algorithm and workspace on the immutable plan before
    measured execution begins.

`qwen35_executor`:

- Replaced per-use `unique_ptr` plan ownership with shared cached plans.
- Added `PlanCacheEntry` keyed by shape, dtype, layout order, transposition, and
  workspace budget.
- On cache miss, creates the plan and tunes only if the bucket is `1`, `128`, or
  `512`.
- Reuses selected plan metadata for identical projection shapes.
- Uses existing RAFT/RMM-backed workspace buffers only; no decode-time
  allocation path was added.
- Tuning is called from executor construction before any Task 6 graph capture.
- Added test-only construction selection diagnostics for algorithm IDs and
  selection-call counts.

## GREEN Verification

Host checks:

```bash
scripts/local-check.sh
```

Result:

```text
100% tests passed, 0 tests failed out of 18
Rust tests: brt-cli 3 passed; cli 14 passed; brt-runtime 0 passed;
engine 14 passed; tokenizer 18 passed; brt-sys 0 passed; doc-tests passed.
```

Whitespace:

```bash
git diff --check
```

Result: exit 0, no output.

CUDA target checks unavailable locally:

```bash
cmake -S . -B build/cuda -G Ninja -DBRT_ENABLE_CUDA=ON -DBRT_BUILD_TESTS=ON
```

Result: fails before target generation because `nvcc` is not installed on this
host.

## Self-Review

- Tuning is construction-only: only `create_plans()` calls `select_fastest`.
- `run_matmul` and `CublasLtMatmulPlan::run` do not enumerate or select.
- Decode graph capture remains after executor construction.
- The existing materialized reference path is untouched.
- No new dependencies were added.
- No `.omx/plans/` files were changed.
- Only one unresolved verification gap remains: CUDA target compile/runtime on
  RTX 50 `sm_120a`.

## Target-Only Verification Gaps

- Could not build `brt_cublaslt_matmul_test` or `brt_qwen35_executor_test`
  because no CUDA toolkit is installed locally.
- Could not run GPU correctness/performance smoke because there is no local
  `nvidia-smi`/RTX 50 GPU and the repo CUDA dev image is absent.
- Target owner should run:

```bash
cmake -S . -B build/cuda -G Ninja -DBRT_ENABLE_CUDA=ON -DBRT_BUILD_TESTS=ON
cmake --build build/cuda --target brt_cublaslt_matmul_test brt_qwen35_executor_test
BRT_RUN_GPU_TESTS=1 ctest --test-dir build/cuda --output-on-failure \
  -R 'brt_cublaslt_matmul_test|brt_qwen35_executor_test|brt_qwen35_cuda_graph_test'
scripts/gpu-smoke.sh
git diff --check
```

## Fix Round 1

### Findings Addressed

1. Renamed the anonymous helper `algorithm_id(...)` to
   `cublaslt_algorithm_config_id(...)` to avoid collision with
   `CublasLtMatmulPlan::algorithm_id()`.
2. Hard-capped `enumerate_cublaslt_candidates` to request at most 16 heuristics
   regardless of a larger public `maximum` argument, and added a behavioral
   oversized-request test.
3. Added production `Qwen35ExecutionDiagnostics::cublaslt_algorithm_ids`.
   Values are populated from immutable construction-time cached plans on cache
   miss. The executor test now asserts the IDs through `executor.diagnostics()`
   and verifies they remain stable across prefill, graph capture, and replay.
4. Changed the graph/tuning fixture from `materialized_policy()` to the
   release-shape online path (`Qwen35ExecutionPolicy{}` with
   `decode_graph = true`) so graph capture/replay assertions are valid.
5. Guarded construction-time tuning with `bucket <= max_context_` so an executor
   created for a smaller context does not tune a larger release bucket using
   undersized construction scratch. Plans are still created and their selected
   first-heuristic IDs are recorded in diagnostics.

### Fix Round 1 Verification

Host checks:

```bash
scripts/local-check.sh
```

Output summary:

```text
100% tests passed, 0 tests failed out of 18
Rust tests: brt-cli 3 passed; cli 14 passed; brt-runtime 0 passed;
engine 14 passed; tokenizer 18 passed; brt-sys 0 passed; doc-tests passed.
```

Whitespace:

```bash
git diff --check
```

Result: exit 0, no output.

Target CUDA/NVCC/GPU rerun: intentionally left to controller per Fix Round 1
brief. Local host still has no `nvcc`/RTX 50 GPU.

## Fix Round 2

### Findings Addressed

1. Repaired `run_cublaslt_plan_tuning_tests` so the release-shape fixture uses
   `qwen35.context_length = 512` with `max_context = 512`. The test now
   directly asserts tuned construction metadata for release buckets 1, 128, and
   512.
2. Added production `Qwen35CublasLtPlanDiagnostic` metadata to executor
   diagnostics: bucket, shape, tuned flag, selected algorithm ID, and workspace
   bytes. The test asserts diagnostic stability across prefill, decode graph
   capture, and replay, proving runtime execution does not reselect plans.
3. Repaired the CUDA graph oracle so independently autotuned ordinary/graph
   executors are compared with numerical tolerance, while exact bitwise
   capture-vs-replay observations are checked on the same graph executor after
   reset with the same selected plans.
4. Addressed the deferred `execution_calls` gap behaviorally through production
   selection metadata: construction diagnostics are snapshotted and asserted
   unchanged after prefill/decode/capture/replay.

Commit: `5a161c528ba22c52c4208528250bcdaec4a23d17`

### Fix Round 2 Verification

Whitespace:

```bash
git diff --check
git diff --check HEAD~1 HEAD
```

Result: both exited 0 with no output.

Host checks:

```bash
scripts/local-check.sh
```

Output summary:

```text
100% tests passed, 0 tests failed out of 18
Rust tests: brt-cli 3 passed; cli 14 passed; brt-runtime 0 passed;
engine 14 passed; tokenizer 18 passed; brt-sys 0 passed; doc-tests passed.
```

Target CUDA/NVCC/GPU rerun remains delegated to the controller. Local host still
has no `nvcc`/RTX 50 GPU, so the focused CUDA targets were not compiled or run
locally.

## Fix Round 3

### Findings Addressed

1. Removed the independent ordinary executor from
   `run_executor_graph_equivalence_tests`. The graph test no longer compares
   logits or tokens across independently autotuned executor constructions.
2. Removed the 2% cross-executor tolerance helper instead of widening it. That
   comparison was unrelated to graph correctness and was already shown flaky by
   the target repeat gate.
3. Preserved the exact same-executor oracle: the first graph sequence records
   result, logits, KV cache, convolution state, and recurrent state; after
   `reset()`, the replay sequence must exactly match those observations.
4. Kept graph diagnostics assertions for initial capture/no replay, reset-time
   immediate replay, later replay, and stable construction-selected cuBLASLt
   plan diagnostics across decode execution.

Commit: `243e5bca71d090fb4d1add144bf0217792886ca2`

### Fix Round 3 Verification

Whitespace:

```bash
git diff --check
git diff --check HEAD~1 HEAD
```

Result: both exited 0 with no output.

Host checks:

```bash
scripts/local-check.sh
```

Output summary:

```text
100% tests passed, 0 tests failed out of 18
Rust tests: brt-cli 3 passed; cli 14 passed; brt-runtime 0 passed;
engine 14 passed; tokenizer 18 passed; brt-sys 0 passed; doc-tests passed.
```

Target CUDA/NVCC/GPU repeat gate remains delegated to the controller. Local host
still has no `nvcc`/RTX 50 GPU.

## Final Target Evidence

Commit validated: `243e5bca71d090fb4d1add144bf0217792886ca2`

- CUDA 13.2 `sm_120a` compile passed.
- Focused GPU tests passed 3/3:
  `brt_cublaslt_matmul_test`, `brt_qwen35_executor_test`, and
  `brt_qwen35_cuda_graph_test`.
- Graph repeat gate passed: `ctest --repeat until-fail:12` completed 12/12
  successful runs after Fix Round 3.
- Independent review approved.
- Real Qwen3.5-9B BF16 parity passed 4/4 exact greedy-token prompts with 32
  generated tokens each. Evidence:
  `/home/charles/brt-validation/task7-243e5bc/qwen35-task7-parity.jsonl`.
- Post-verification preflight was idle: 31972 MiB free, 0% utilization, 35C.

Task 7 is complete.
