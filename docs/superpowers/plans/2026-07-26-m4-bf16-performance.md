# M4 BF16 Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the BF16 PP512 and TG128@PP512 performance deficits while preserving exact Qwen3.5-9B greedy-token parity.

**Architecture:** Keep the existing materialized attention as an independent diagnostic reference and add a project-owned tiled online-softmax implementation selected by an immutable execution policy. Reduce duplicate FP32-to-BF16 conversion launches without changing arithmetic, then capture the fixed-address decode path in a CUDA Graph. Only promote BF16 KV cache or additional matrix paths after their independent correctness and performance gates pass.

**Tech Stack:** C++20, CUDA 13, cuBLASLt, RAFT 26.06, RMM 26.06, CMake/Ninja, Rust 2024, Bash, CTest.

## Global Constraints

- Target RTX 50-series `sm_120a` only; RTX 40-series support is excluded.
- RAFT/RMM remain the common stream, allocation, and workspace foundation.
- Rust submits one prefill or decode step per FFI call; no per-operator FFI is introduced.
- The fixed four-prompt corpus must retain exact greedy-token equality for 32 generated tokens per prompt.
- CUDA operators must satisfy the existing `abs <= 2e-2 || rel <= 2e-2` FP32-reference contract unless a tighter operator-specific limit already exists.
- BF16 release gates are PP128 `>=1.0x`, PP512 `>=1.0x`, and TG128 after PP512 `>=1.0x` versus pinned same-configuration llama.cpp, with at least one metric `>=1.1x`.
- Performance evidence is invalid when another compute process occupies the target GPU.
- No decode-time RMM allocation is allowed.
- Existing materialized attention remains available only as a correctness and diagnostic reference.

---

## Planned File Structure

```text
cpp/
├── execution/
│   ├── cuda_graph_decode.hpp/.cu       # graph ownership, capture, replay, reset
│   ├── qwen35_execution_policy.hpp     # immutable optimized/reference choices
│   ├── qwen35_executor.cu              # plan integration and grouped matmuls
│   └── cublaslt_matmul.hpp/.cu         # candidate enumeration and cached choice
├── kernels/
│   ├── qwen35_attention.cuh/.cu        # reference path and public dispatch
│   ├── qwen35_online_attention.cuh/.cu # tiled prefill/decode implementation
│   └── qwen35_delta.cu                 # measured launch policy only
└── tests/
    ├── qwen35_cuda_attention_test.cu
    ├── qwen35_cuda_delta_test.cu
    ├── qwen35_cuda_graph_test.cu
    ├── qwen35_executor_test.cu
    └── cublaslt_matmul_test.cu
scripts/
├── qwen35-benchmark.sh
└── qwen35-bf16-gate.sh
tests/
└── bf16-gate-script-test.sh
docs/
└── m4-verification.md
```

### Task 1: Lock the execution-policy and attention-dispatch contract

**Files:**
- Create: `cpp/execution/qwen35_execution_policy.hpp`
- Modify: `cpp/kernels/qwen35_attention.cuh`
- Modify: `cpp/kernels/qwen35_attention.cu`
- Modify: `cpp/tests/qwen35_cuda_attention_test.cu`

**Interfaces:**
- Produces:

```cpp
namespace raftinfer {
enum class Qwen35AttentionImplementation : std::uint8_t {
  materialized_reference,
  online_tiled,
};

enum class Qwen35KvCacheDType : std::uint8_t {
  f32,
  bf16,
};

enum class Qwen35KvCacheLayout : std::uint8_t {
  token_major,
  head_major,
};

struct Qwen35ExecutionPolicy {
  Qwen35AttentionImplementation attention{
      Qwen35AttentionImplementation::online_tiled};
  Qwen35KvCacheDType kv_cache{Qwen35KvCacheDType::f32};
  Qwen35KvCacheLayout kv_cache_layout{
      Qwen35KvCacheLayout::token_major};
  bool decode_graph{true};
  bool grouped_input_casts{true};
};
}

namespace raftinfer::kernels {
struct Qwen35AttentionLaunchPolicy {
  raftinfer::Qwen35AttentionImplementation implementation;
  raftinfer::Qwen35KvCacheDType kv_cache_dtype;
  raftinfer::Qwen35KvCacheLayout kv_cache_layout;
};
}
```

- Changes the attention API to accept `Qwen35AttentionLaunchPolicy` and byte-sized
  cache/workspace spans rather than assuming `float*`.

- [ ] **Step 1: Write compile-time and runtime contract tests**

Add tests that require both attention implementations, reject a BF16 cache with
an incorrectly sized buffer, and verify that the reference policy still
produces the existing output:

```cpp
const raftinfer::kernels::Qwen35AttentionLaunchPolicy reference{
    .implementation =
        raftinfer::Qwen35AttentionImplementation::materialized_reference,
    .kv_cache_dtype = raftinfer::Qwen35KvCacheDType::f32,
};
assert(raftinfer::kernels::qwen35_attention_cache_bytes(shape, reference) ==
       2 * shape.max_context_tokens * shape.kv_heads * shape.head_dim *
           sizeof(float));
assert(raftinfer::kernels::qwen35_attention_workspace_bytes(shape, reference) ==
       shape.tokens * shape.query_heads *
           (shape.past_tokens + shape.tokens) * sizeof(float));
```

- [ ] **Step 2: Run the CUDA attention test and verify RED**

Run on the RTX 50 target:

```bash
cmake --build <build-root>/m4/cuda --target raftinfer_qwen35_cuda_attention_test
ctest --test-dir <build-root>/m4/cuda \
  -R '^raftinfer_qwen35_cuda_attention_test$' --output-on-failure
```

Expected: compilation fails because the policy types and byte-sized API do not
exist.

- [ ] **Step 3: Add the policy and preserve the old implementation**

Move the existing implementation behind:

```cpp
void qwen35_causal_attention_materialized(
    const void* query, const void* key, const void* value, const void* gate,
    void* output, void* kv_cache, std::size_t kv_cache_bytes,
    void* workspace, std::size_t workspace_bytes, Qwen35AttentionShape shape,
    RaftInferDataType activation_dtype, Qwen35KvCacheDType cache_dtype,
    cudaStream_t stream);
```

The public `qwen35_causal_attention` dispatches only after validating every
pointer, byte span, dtype, and shape. The materialized implementation rejects
non-F32 cache dtype.

- [ ] **Step 4: Run the focused test and host checks**

Run:

```bash
ctest --test-dir <build-root>/m4/cuda \
  -R '^raftinfer_qwen35_cuda_attention_test$' --output-on-failure
scripts/local-check.sh
```

Expected: CUDA attention test and all host checks pass with unchanged reference
outputs.

- [ ] **Step 5: Commit**

```bash
git add cpp/execution/qwen35_execution_policy.hpp \
  cpp/kernels/qwen35_attention.cuh cpp/kernels/qwen35_attention.cu \
  cpp/tests/qwen35_cuda_attention_test.cu
git commit -m "refactor: add Qwen3.5 attention execution policy"
```

### Task 2: Implement tiled online-softmax prefill attention

**Files:**
- Create: `cpp/kernels/qwen35_online_attention.cuh`
- Create: `cpp/kernels/qwen35_online_attention.cu`
- Modify: `cpp/kernels/qwen35_attention.cu`
- Modify: `cpp/tests/qwen35_cuda_attention_test.cu`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Consumes: `Qwen35AttentionShape`, `Qwen35AttentionLaunchPolicy`.
- Produces:

```cpp
std::size_t qwen35_online_attention_workspace_bytes(
    Qwen35AttentionShape shape) noexcept;

void qwen35_online_attention_prefill(
    const void* query, const void* key, const void* value, const void* gate,
    void* output, void* kv_cache, std::size_t kv_cache_bytes,
    Qwen35AttentionShape shape, RaftInferDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype, cudaStream_t stream);
```

- [ ] **Step 1: Add failing parity cases**

For F32 and BF16 activations, compare online and materialized outputs for:

```text
tokens=4,  heads=4,  kv_heads=2, head_dim=64,  past=0
tokens=17, heads=16, kv_heads=4, head_dim=256, past=0
tokens=128,heads=16, kv_heads=4, head_dim=256, past=0
tokens=17, heads=16, kv_heads=4, head_dim=256, past=111
```

Require finite output, causal isolation, `max_abs <= 2e-2`, and
`max_rel <= 2e-2` for BF16. Add an adversarial fixture with logits near
`[-80, 80]` to prove stable online rescaling.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: link failure for `qwen35_online_attention_prefill`.

- [ ] **Step 3: Implement the online recurrence**

Use a `Q_TILE=4`, `K_TILE=16`, four-warp block for the model shape. Each warp
owns one query and each lane owns eight output dimensions for `head_dim=256`.
For every visible key tile, update:

```cpp
const float next_max = fmaxf(row_max, tile_max);
const float old_scale = expf(row_max - next_max);
const float tile_scale = expf(score - next_max);
row_sum = row_sum * old_scale + tile_scale;
for (int component = 0; component < kValuesPerLane; ++component) {
  out_acc[component] =
      out_acc[component] * old_scale + tile_scale * value_component[component];
}
row_max = next_max;
```

The kernel writes no logits tensor. Q/K/V inputs may be BF16 or F32; online
state and output accumulation remain FP32. Gate application occurs before the
final typed store.

- [ ] **Step 4: Add model-shape launch specialization**

Instantiate explicit `head_dim=256`, `query_heads=16`, `kv_heads=4` kernels
with:

```cpp
__launch_bounds__(128, 2)
```

Retain a checked generic online path for test dimensions. Unsupported release
shapes fall back to the materialized reference and report that choice.

- [ ] **Step 5: Run tests**

Run the attention test, full CUDA CTest, and `git diff --check`.

Expected: all numerical cases pass and the optimized policy reports zero logits
workspace bytes.

- [ ] **Step 6: Commit**

```bash
git add cpp/kernels/qwen35_online_attention.cuh \
  cpp/kernels/qwen35_online_attention.cu cpp/kernels/qwen35_attention.cu \
  cpp/tests/qwen35_cuda_attention_test.cu cpp/CMakeLists.txt
git commit -m "feat: add tiled online-softmax prefill attention"
```

### Task 3: Implement long-context online decode attention

**Files:**
- Modify: `cpp/kernels/qwen35_online_attention.cuh`
- Modify: `cpp/kernels/qwen35_online_attention.cu`
- Modify: `cpp/kernels/qwen35_attention.cu`
- Modify: `cpp/tests/qwen35_cuda_attention_test.cu`

**Interfaces:**
- Produces:

```cpp
void qwen35_online_attention_decode(
    const void* query, const void* key, const void* value, const void* gate,
    void* output, void* kv_cache, std::size_t kv_cache_bytes,
    Qwen35AttentionShape shape, RaftInferDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype, const std::uint32_t* device_position,
    cudaStream_t stream);
```

- [ ] **Step 1: Add failing decode comparisons**

Compare online decode with materialized attention at context lengths
`32, 128, 512, 2048`, including GQA head mapping and nonzero cache contents.
Execute two consecutive decode calls and verify cache append ordering.

- [ ] **Step 2: Verify RED**

Expected: compilation fails because the decode entry point and device-position
form are absent.

- [ ] **Step 3: Implement split-context online reduction**

Use one block per query head. Warps process disjoint key ranges and maintain
local `(max, sum, output)` states. Merge warp states with:

```cpp
const float merged_max = fmaxf(left.max, right.max);
const float left_scale = expf(left.max - merged_max);
const float right_scale = expf(right.max - merged_max);
merged.sum = left.sum * left_scale + right.sum * right_scale;
merged.output =
    left.output * left_scale + right.output * right_scale;
```

For `context <= 128`, use four warps; for `context > 128`, use eight warps.
Read the logical position from `device_position` when non-null and from
`shape.past_tokens` for ordinary prefill/diagnostic execution.

- [ ] **Step 4: Verify correctness and launch metadata**

Require one fused decode-attention kernel plus one cache-append kernel before
graph work. The test records the implementation name
`qwen35_online_attention_decode_sm120_hd256`.

- [ ] **Step 5: Commit**

```bash
git add cpp/kernels/qwen35_online_attention.cuh \
  cpp/kernels/qwen35_online_attention.cu cpp/kernels/qwen35_attention.cu \
  cpp/tests/qwen35_cuda_attention_test.cu
git commit -m "feat: add online Qwen3.5 decode attention"
```

### Task 4: Integrate online attention and remove optimized logits workspace

**Files:**
- Modify: `cpp/execution/qwen35_executor.hpp`
- Modify: `cpp/execution/qwen35_executor.cu`
- Modify: `cpp/tests/qwen35_executor_test.cu`
- Modify: `cpp/tests/workspace_layout_test.cpp`

**Interfaces:**
- `Qwen35Executor` constructor gains:

```cpp
Qwen35Executor(
    ExecutionContext& context, const model::Qwen35Config& config,
    const model::CudaWeightPlan& weights, std::size_t max_context,
    Qwen35ExecutionPolicy policy = {});
```

- Produces read-only diagnostics:

```cpp
struct Qwen35ExecutionDiagnostics {
  Qwen35AttentionImplementation attention;
  Qwen35KvCacheDType kv_cache_dtype;
  Qwen35KvCacheLayout kv_cache_layout;
  bool decode_graph_captured;
  bool decode_graph_replayed;
  std::size_t attention_workspace_bytes;
};
```

- [ ] **Step 1: Add failing executor policy tests**

Construct reference and online executors from the same fixture. Require equal
tokens, matching positions, output error within the existing limit, and:

```cpp
assert(online.diagnostics().attention_workspace_bytes == 0);
assert(reference.diagnostics().attention_workspace_bytes > 0);
```

- [ ] **Step 2: Verify RED**

Expected: constructor and diagnostics compilation failures.

- [ ] **Step 3: Thread policy through allocation and launch**

Compute cache/workspace sizes through the attention API. Allocate no logits
workspace for the online policy. Preserve fixed addresses and the reference
policy for localization.

- [ ] **Step 4: Run executor and no-allocation tests**

Require repeated prefill/decode to keep the RMM peak unchanged after session
construction for both policies.

- [ ] **Step 5: Commit**

```bash
git add cpp/execution/qwen35_executor.hpp cpp/execution/qwen35_executor.cu \
  cpp/tests/qwen35_executor_test.cu cpp/tests/workspace_layout_test.cpp
git commit -m "feat: select online attention in Qwen3.5 executor"
```

### Task 5: Deduplicate mathematically identical input casts

**Files:**
- Modify: `cpp/execution/qwen35_executor.cu`
- Modify: `cpp/tests/qwen35_executor_test.cu`

**Interfaces:**
- Produces:

```cpp
struct MatmulBinding {
  const CublasLtMatmulPlan* plan;
  const model::CudaTensorView* weight;
  void* output;
};

void run_matmul_group(
    const void* f32_input, std::span<const MatmulBinding> bindings);
```

- [ ] **Step 1: Add a launch-accounting test seam**

Add an internal counter incremented by `qwen35_cast_f32` calls when test
instrumentation is enabled. For one decode layer require:

```text
full attention projection group: one cast for query/key/value
linear attention projection group: one cast for qkv/beta/alpha/gate
FFN gate/up group: one cast
```

The outputs must remain byte-identical to the ungrouped conversion path because
the converted input bytes and cuBLASLt operations are unchanged.

- [ ] **Step 2: Verify the test fails**

Expected: existing implementation reports three, four, and two casts for the
respective groups.

- [ ] **Step 3: Implement grouped conversion**

Validate that every binding has identical input bytes/dtype, cast once into
`matmul_input_`, then issue the plans sequentially before reusing the buffer.
Do not group operations whose input tensor differs.

- [ ] **Step 4: Run full executor parity**

Run CUDA executor tests plus the four-prompt real-model parity script. Expected:
exact token IDs remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add cpp/execution/qwen35_executor.cu cpp/tests/qwen35_executor_test.cu
git commit -m "perf: reuse Qwen3.5 projection input casts"
```

### Task 6: Capture and replay the fixed-address decode graph

**Files:**
- Create: `cpp/execution/cuda_graph_decode.hpp`
- Create: `cpp/execution/cuda_graph_decode.cu`
- Modify: `cpp/execution/qwen35_executor.hpp`
- Modify: `cpp/execution/qwen35_executor.cu`
- Create: `cpp/tests/qwen35_cuda_graph_test.cu`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
class CudaGraphDecode {
public:
  using CaptureBody = std::function<void()>;

  CudaGraphDecode(int device_id, cudaStream_t stream);
  ~CudaGraphDecode() noexcept;
  void capture(CaptureBody body);
  void replay();
  void reset() noexcept;
  bool captured() const noexcept;
};
```

- Executor owns fixed pinned host token/result scalars plus device token,
  position, and result scalars.

- [ ] **Step 1: Add generic graph RAII tests**

Capture a two-kernel integer transform, replay with different pinned input
values, verify output, reset, recapture, and verify all CUDA graph handles are
destroyed on failure.

- [ ] **Step 2: Add failing executor graph-equivalence tests**

Run ordinary and graph sessions through prefill plus eight decode steps. Require
equal token, position, recurrent state, KV cache, and logits diagnostics after
each step. Reset both and repeat.

- [ ] **Step 3: Implement graph RAII**

Use:

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
body();
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
cudaGraphLaunch(exec, stream);
```

Every CUDA failure destroys partially created resources and leaves
`captured()==false`.

- [ ] **Step 4: Make decode position graph-compatible**

Pass a device position pointer to RoPE and online decode attention. Capture
fixed-address H2D token copy, complete decode, D2H result copy, and a final
device position increment. `reset()` writes zero to the device position and
invalidates no topology.

- [ ] **Step 5: Integrate lazy capture**

The first optimized decode materializes cuBLAS kernels and executes ordinarily.
Capture occurs only after all lazy library loads have completed. Later calls
replay the graph and set `diagnostics.decode_graph_replayed=true`.

- [ ] **Step 6: Run graph, executor, and parity suites**

Expected: ordinary and graph outputs match, no decode allocations occur, and
the four-prompt greedy-token corpus remains exact.

- [ ] **Step 7: Commit**

```bash
git add cpp/execution/cuda_graph_decode.hpp \
  cpp/execution/cuda_graph_decode.cu cpp/execution/qwen35_executor.hpp \
  cpp/execution/qwen35_executor.cu cpp/tests/qwen35_cuda_graph_test.cu \
  cpp/CMakeLists.txt
git commit -m "feat: replay Qwen3.5 decode with CUDA Graph"
```

### Task 7: Tune cached matrix algorithms for token shape

**Files:**
- Modify: `cpp/execution/cublaslt_matmul.hpp`
- Modify: `cpp/execution/cublaslt_matmul.cu`
- Modify: `cpp/execution/qwen35_executor.cu`
- Modify: `cpp/tests/cublaslt_matmul_test.cu`

**Interfaces:**
- Produces:

```cpp
struct CublasLtCandidate {
  cublasLtMatmulAlgo_t algorithm;
  int algorithm_id;
  std::size_t workspace_bytes;
};

std::vector<CublasLtCandidate> enumerate_cublaslt_candidates(
    const CublasLtMatmulConfig& config, std::size_t maximum = 16);

void CublasLtMatmulPlan::select_fastest(
    cudaStream_t stream, const void* input, const void* weight, void* output,
    void* workspace, std::size_t workspace_bytes,
    std::uint32_t warmups = 2, std::uint32_t measurements = 5);
```

- [ ] **Step 1: Add candidate and deterministic-selection tests**

Require candidate enumeration to return only successful algorithms within the
workspace budget. Compare every selected result with the original first
heuristic output.

- [ ] **Step 2: Verify RED**

Expected: missing enumeration and selection APIs.

- [ ] **Step 3: Enumerate and time candidates once**

Request up to 16 heuristics, reject unsuccessful or oversized candidates, time
with CUDA events during executor construction, and retain the lowest median.
Never time or select algorithms inside prefill/decode.

- [ ] **Step 4: Restrict tuning to release shapes**

Tune bucket `1`, `128`, and `512` plans. Cache the selected algorithm ID in
diagnostics. If multiple projections share identical shape/dtype/transposition,
reuse the selected algorithm metadata.

- [ ] **Step 5: Run correctness and performance smoke**

Expected: identical matrix outputs and no plan selection calls during measured
execution.

- [ ] **Step 6: Commit**

```bash
git add cpp/execution/cublaslt_matmul.hpp \
  cpp/execution/cublaslt_matmul.cu cpp/execution/qwen35_executor.cu \
  cpp/tests/cublaslt_matmul_test.cu
git commit -m "perf: tune cuBLASLt plans by token shape"
```

### Task 8: Tune the retained register-resident Gated DeltaNet schedules

**Files:**
- Modify: `cpp/kernels/qwen35_delta.cuh`
- Modify: `cpp/kernels/qwen35_delta.cu`
- Modify: `cpp/execution/qwen35_executor.cu`
- Modify: `cpp/tests/qwen35_cuda_delta_test.cu`
- Modify: `cpp/tests/qwen35_executor_test.cu`

**Interfaces:**
- Produces:

```cpp
enum class GatedDeltaSchedule : std::uint8_t {
  register_resident_current,
  register_resident_prefill_sm120,
  register_resident_decode_sm120,
};

struct GatedDeltaLaunchPolicy {
  GatedDeltaSchedule schedule;
  std::uint32_t warps_per_block;
  bool transposed_boundary_state;
};
```

- [x] **Step 1: Lock current-kernel parity and timing fixtures**

For key/value dimensions `64` and `128`, compare prefill lengths
`17,128,512`, continued prefill, and eight single-token decode steps against
the existing CPU FP32 reference. Snapshot convolution state, recurrent state,
and hidden output after every call. Add a microbenchmark that reports the
current kernel median separately for prefill and decode.

- [x] **Step 2: Run the focused tests before optimization**

```bash
ctest --test-dir <build-root>/m4/cuda \
  -R '^raftinfer_qwen35_cuda_delta_test$' --output-on-failure
```

Expected: the current register-resident schedule passes and establishes the
candidate-selection baseline.

- [x] **Step 3: Add explicit RTX 50 candidate schedules**

Implement four-warps-per-block column ownership, vectorized coalesced boundary
loads/stores, and explicit dimension specializations. Use:

```cpp
template <int KeyDim, int ValueDim, bool Decode>
__global__ __launch_bounds__(128, 2)
void gated_delta_register_kernel(GatedDeltaKernelArgs args);
```

The prefill variant keeps state shards in registers across its token loop. The
decode variant removes loop setup that is invariant for one token. A transposed
boundary-state layout is eligible only when conversion occurs during session
initialization/reset, not during measured decode.

- [x] **Step 4: Select only a faster passing schedule**

At immutable plan construction, benchmark current and candidate schedules for
the exact model dimensions and token buckets `1,128,512`. Reject any candidate
whose output/state comparison fails. Keep
`register_resident_current` when no candidate improves the median; do not
replace it merely because the new schedule is structurally closer to llama.cpp.

- [x] **Step 5: Run executor parity and real-model parity**

Expected: operator/state tests pass, selected schedule metadata is stable, no
decode allocation is introduced, and all four prompts retain exact generated
token IDs.

Final Task 8 commit `ffc99ed` passed target CUDA 13.2 `sm_120a` compile and
focused GPU CTest 3/3: delta 0.51s, executor 0.54s, graph 1.71s. The focused
microbenchmark evidence is
`<artifact-root>/task8-ffc99ed/task8-delta-microbench.jsonl`:
bucket `1` current/candidate medians 0.014368/0.008192 ms, bucket `128`
0.607232/0.14336 ms, and bucket `512` 2.40742/0.548864 ms; all candidates were
correctness-passing and accepted. Independent review is APPROVE. Real
Qwen3.5-9B parity passed 4/4 prompts × 32 exact tokens at
`<artifact-root>/task8-ffc99ed/qwen35-task8-parity.jsonl`, and
post-validation preflight reported the GPU idle. The 64-dim fixture
microbenchmark is a focused schedule-selection diagnostic, not end-to-end
throughput evidence; the real 128-dim model path is covered by exact parity.

- [x] **Step 6: Commit**

```bash
git add cpp/kernels/qwen35_delta.cuh cpp/kernels/qwen35_delta.cu \
  cpp/execution/qwen35_executor.cu cpp/tests/qwen35_cuda_delta_test.cu \
  cpp/tests/qwen35_executor_test.cu
git commit -m "perf: tune Qwen3.5 Gated DeltaNet schedules"
```

### Task 9: Add BF16 KV-cache dtype and layout A/B mode

**Files:**
- Modify: `cpp/kernels/qwen35_online_attention.cu`
- Modify: `cpp/execution/qwen35_executor.cu`
- Modify: `cpp/tests/qwen35_cuda_attention_test.cu`
- Modify: `cpp/tests/qwen35_executor_test.cu`

**Interfaces:**
- Consumes: `Qwen35KvCacheDType`, `Qwen35KvCacheLayout`.
- Produces vectorized BF16 cache append/load while retaining FP32 accumulation.

- [ ] **Step 1: Add cache conversion and state-transition tests**

Run identical prefill, continued prefill, decode, and reset sequences with F32
and BF16 caches under both token-major and head-major layouts. Compare attention
outputs to the materialized F32 reference and require identical greedy argmax
in the executor fixture.

- [ ] **Step 2: Verify RED**

Expected: optimized attention rejects BF16 cache dtype.

- [ ] **Step 3: Implement BF16 cache access**

Store cache elements with `__float2bfloat16_rn`, load with
`__bfloat162float`, and use `__nv_bfloat162` vector operations when addresses
are four-byte aligned. Implement explicit token-major and head-major index
functions and keep online max/sum/output in FP32.

- [ ] **Step 4: Benchmark layouts and run real-model parity before promotion**

Benchmark both layouts at PP512 and TG128@PP512, then run the four-prompt,
32-token corpus under the fastest passing mode. BF16 becomes the default only
if exact token IDs match and PP512 or TG128 improves without regressing the
other below its gate. Otherwise retain the F32 token-major release path and
report the rejected BF16/layout candidates.

- [ ] **Step 5: Commit**

```bash
git add cpp/kernels/qwen35_online_attention.cu \
  cpp/execution/qwen35_executor.cu \
  cpp/tests/qwen35_cuda_attention_test.cu cpp/tests/qwen35_executor_test.cu
git commit -m "perf: add validated BF16 Qwen3.5 KV cache"
```

### Task 10: Enforce and document the BF16 performance gate

**Files:**
- Create: `scripts/qwen35-bf16-gate.sh`
- Create: `tests/bf16-gate-script-test.sh`
- Modify: `scripts/qwen35-benchmark.sh`
- Modify: `scripts/local-check.sh`
- Create: `docs/verification/m4.md`

**Interfaces:**
- Consumes the two JSONL benchmark arms from `qwen35-benchmark.sh`.
- Produces exit `0` only when:

```jq
(.prompt_tokens == 128 and .throughput_ratio.prefill >= 1.0) or
(.prompt_tokens == 512 and
 .throughput_ratio.prefill >= 1.0 and
 .throughput_ratio.generation >= 1.0)
```

and the maximum of PP128, PP512, TG128@PP512 ratios is at least `1.1`.

- [ ] **Step 1: Write shell fixtures for pass and each failure**

Cover PP128 below 1.0, PP512 below 1.0, TG128 below 1.0, no metric at 1.1,
wrong dtype, missing graph replay, incorrect token parity, and unstable
coefficient of variation.

- [ ] **Step 2: Run and verify RED**

Run `tests/bf16-gate-script-test.sh`. Expected: missing gate script.

- [ ] **Step 3: Implement strict JSONL validation**

Require BF16 weight format, selected online attention, disclosed KV dtype,
graph replay for TG128, pinned llama revision, three or more warmups, seven or
more measurements, and coefficient of variation no greater than three percent.

- [ ] **Step 4: Run local and target gates**

Run:

```bash
scripts/local-check.sh
scripts/gpu-preflight.sh
scripts/qwen35-parity.sh
scripts/qwen35-benchmark.sh
scripts/qwen35-bf16-gate.sh build/evidence/qwen35-benchmark.jsonl
```

Expected: all local checks, 23+ CUDA tests, exact parity, and every BF16
performance gate pass.

- [ ] **Step 5: Record verification**

Write exact commit, binary/model hashes, selected kernels, cache dtype, graph
status, peak RMM bytes, PP128, PP512, and TG128@PP512 results to
`docs/verification/m4.md`.

- [ ] **Step 6: Commit**

```bash
git add scripts/qwen35-bf16-gate.sh scripts/qwen35-benchmark.sh \
  scripts/local-check.sh tests/bf16-gate-script-test.sh \
  docs/verification/m4.md
git commit -m "test: enforce BF16 performance release gate"
```
