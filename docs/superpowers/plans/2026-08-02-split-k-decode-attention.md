# Split-K Decode Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a graph-compatible, context-bucketed split-K online-attention decode path for Qwen3.5-9B BF16 that improves long-context generation on RTX 5090 without weakening numerical or exact-token correctness.

**Architecture:** Keep the current one-block-per-query-head kernel as the short-context and unsupported-signature path. Add a first-stage partition kernel that writes FP32 online-softmax partials and a second-stage per-head merge kernel, with fixed RMM-owned scratch and one CUDA Graph variant per context bucket. Resolve `auto`, `single-block`, `split-k-256`, and `split-k-512` once at session construction before workspace sizing, expose the resolved path through diagnostics, and promote a split-K mode only after two independent target runs pass the approved baseline-relative gate.

**Tech Stack:** C++20, CUDA 13, RAFT 26.06, RMM 26.06, CMake/Ninja, Rust 2024, Bash, jq, CTest.

## Global Constraints

- Target RTX 50-series `sm_120a` only; RTX 40-series support remains excluded.
- The release shape is Qwen3.5-9B BF16 with 16 query heads, 4 KV heads, head dimension 256, and BF16 head-major KV cache.
- RAFT/RMM remain the stream, allocation, and fixed-workspace foundation.
- Rust continues to submit complete prefill, decode-step, or graph-replay operations; no per-operator FFI is introduced.
- Public Rust request method signatures and sampling semantics remain unchanged.
- Short-context online decode remains zero-workspace; split-K scratch is fixed-address, bounded, and allocated before graph capture.
- CUDA decode performs no allocation, resize, descriptor construction, or algorithm search after execution-plan construction; runtime dispatch is limited to an O(1) lookup among prebuilt context-bucket variants.
- Operator output must satisfy the existing `abs <= 2e-2 || rel <= 2e-2` reference contract, and all 128 accepted greedy tokens must match exactly.
- Default promotion requires two independent uncontended RTX 5090 runs in which PP512 generation and TG128@PP512 generation each improve by at least 1% over the same-window single-block RAFTInfer baseline.
- PP128 prefill, PP512 prefill, and PP128 generation may not regress by more than 1% from that baseline.
- Split-K that passes correctness but fails the performance gate remains explicitly selectable and is not the `auto` default.
- llama.cpp remains the pinned external performance reference; bw24 code is reused only if compatibility, license, correctness, provenance, and measured-performance gates all pass.

---

## Planned File Structure

```text
cpp/
├── kernels/
│   ├── qwen35_online_attention.cuh     # split-K plan and workspace layout contract
│   └── qwen35_online_attention.cu      # partition, merge, and single-block kernels
├── execution/
│   ├── qwen35_executor.hpp             # read-only split-K diagnostics
│   ├── qwen35_executor.cu              # plan resolution, scratch, graph variants
│   ├── qwen35_execution_policy.hpp     # internal decode-attention diagnostic enum
│   ├── session.hpp
│   └── session.cpp                     # diagnostics propagation
├── include/raftinfer/c_api.h           # additive read-only diagnostic fields
├── src/c_api.cpp                       # diagnostic enum/field mapping
└── tests/
    ├── qwen35_cuda_attention_test.cu   # planner, partial-state, and output parity
    ├── qwen35_cuda_graph_test.cu       # graph bucket and boundary equivalence
    ├── qwen35_executor_test.cu         # scratch and resolved-plan diagnostics
    └── c_api_test.cpp                  # diagnostic ABI mapping
rust/
├── raftinfer-sys/src/lib.rs             # matching diagnostic ABI layout
├── raftinfer-runtime/src/lib.rs         # typed read-only diagnostics
├── raftinfer-runtime/tests/engine.rs
└── raftinfer-cli/src/main.rs             # JSON diagnostic disclosure
scripts/
├── local-check.sh                       # includes the new script gate test
├── qwen35-benchmark.sh                  # require disclosed decode path
└── qwen35-split-k-gate.sh               # same-window two-run promotion gate
tests/
├── benchmark-script-test.sh
└── split-k-gate-script-test.sh
docs/
├── environment.md
├── benchmarks.md
└── verification/raftinfer-release.md
```

## Target Development Convention

The local Mac performs host/Rust/script checks. CUDA RED/GREEN cycles run from
the fixed, project-specific target paths below:

```bash
export RAFTINFER_TARGET='charles@192.168.124.8'
export RAFTINFER_TARGET_DIR='/home/charles/raftinfer-split-k-dev/source'
scripts/sync-target.sh
ssh "${RAFTINFER_TARGET}" \
  'cmake -S /home/charles/raftinfer-split-k-dev/source \
    -B /home/charles/raftinfer-split-k-dev/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DRAFTINFER_ENABLE_CUDA=ON \
    -DRAFTINFER_BUILD_TESTS=ON \
    -DRAFTINFER_NATIVE_LIBRARY_TYPE=SHARED'
```

Run `scripts/sync-target.sh` before each target RED/GREEN build after local
source changes. The development build is disposable and never supplies formal
release evidence.

### Task 1: Lock the split-K planning and workspace contract

**Files:**
- Modify: `cpp/execution/qwen35_execution_policy.hpp:7-29`
- Modify: `cpp/kernels/qwen35_online_attention.cuh:12-35`
- Modify: `cpp/kernels/qwen35_online_attention.cu:16-130,786-919`
- Modify: `cpp/tests/qwen35_cuda_attention_test.cu:203-371`

**Interfaces:**
- Consumes: `Qwen35AttentionShape` and the existing Qwen3.5 online-decode model signature.
- Produces:

```cpp
namespace raftinfer {

enum class Qwen35DecodeAttentionMode : std::uint8_t {
  auto_select,
  single_block,
  split_k_256,
  split_k_512,
};

} // namespace raftinfer

namespace raftinfer::kernels {

struct Qwen35OnlineDecodePlan {
  Qwen35DecodeAttentionMode resolved_mode{
      Qwen35DecodeAttentionMode::single_block};
  std::size_t partition_tokens{};
  std::size_t split_k_threshold_tokens{};
  std::size_t context_bucket_tokens{};
  std::size_t active_partition_capacity{1};

  bool operator==(const Qwen35OnlineDecodePlan &) const = default;
};

struct Qwen35OnlineDecodeWorkspaceLayout {
  std::size_t partial_count{};
  std::size_t max_offset_bytes{};
  std::size_t sum_offset_bytes{};
  std::size_t value_offset_bytes{};
  std::size_t bytes{};

  bool operator==(const Qwen35OnlineDecodeWorkspaceLayout &) const = default;
};

Qwen35OnlineDecodePlan qwen35_online_decode_plan(
    Qwen35AttentionShape shape, Qwen35DecodeAttentionMode requested,
    std::size_t context_tokens);

Qwen35OnlineDecodeWorkspaceLayout qwen35_online_decode_workspace_layout(
    Qwen35AttentionShape shape, Qwen35DecodeAttentionMode resolved_mode);

} // namespace raftinfer::kernels
```

The context bucket is the smallest member of `{512, 1024, 2048, 4096, ...}`
that covers `context_tokens`, capped by `shape.max_context_tokens`. The
single-block plan always has `partition_tokens == 0`, threshold `0`, capacity
`1`, and workspace bytes `0`. Split-K modes use threshold equal to their
partition size and capacity `ceil(context_bucket_tokens / partition_tokens)`.

- [ ] **Step 1: Write failing host-side contract tests**

Add these assertions to `run_attention_policy_contract_tests`:

```cpp
using Mode = raftinfer::Qwen35DecodeAttentionMode;
const raftinfer::kernels::Qwen35AttentionShape model_shape{
    .tokens = 1,
    .query_heads = 16,
    .kv_heads = 4,
    .head_dim = 256,
    .max_context_tokens = 4096,
    .past_tokens = 0,
};

const auto single = raftinfer::kernels::qwen35_online_decode_plan(
    model_shape, Mode::single_block, 128);
assert(single.resolved_mode == Mode::single_block);
assert(single.context_bucket_tokens == 512);
assert(single.active_partition_capacity == 1);
assert(raftinfer::kernels::qwen35_online_decode_workspace_layout(
           model_shape, single.resolved_mode)
           .bytes == 0);

const auto split256 = raftinfer::kernels::qwen35_online_decode_plan(
    model_shape, Mode::split_k_256, 513);
assert(split256.partition_tokens == 256);
assert(split256.split_k_threshold_tokens == 256);
assert(split256.context_bucket_tokens == 1024);
assert(split256.active_partition_capacity == 4);

const auto layout =
    raftinfer::kernels::qwen35_online_decode_workspace_layout(
        model_shape, Mode::split_k_256);
assert(layout.partial_count == 16 * 16);
assert(layout.max_offset_bytes == 0);
assert(layout.sum_offset_bytes == layout.partial_count * sizeof(float));
assert(layout.value_offset_bytes == 2 * layout.partial_count * sizeof(float));
assert(layout.bytes ==
       (2 * layout.partial_count + layout.partial_count * 256) * sizeof(float));
```

Also require structured `Qwen35PrimitiveError` for `context_tokens == 0`,
`context_tokens > max_context_tokens`, and split-K requests on a non-model
signature.

- [ ] **Step 2: Run the CUDA attention target and verify RED**

On the RTX 5090 target source tree, run:

```bash
cmake --build /home/charles/raftinfer-split-k-dev/build \
  --target raftinfer_qwen35_cuda_attention_test
```

Expected: compilation fails because `Qwen35DecodeAttentionMode`, the plan, and the
workspace-layout function do not exist.

- [ ] **Step 3: Implement checked planning and SoA layout arithmetic**

Add overflow-checked helpers in `qwen35_online_attention.cu`. The SoA layout is:

```text
partial_max[query_heads][max_partitions]
partial_sum[query_heads][max_partitions]
partial_value[query_heads][max_partitions][head_dim]
```

`max_partitions` is computed from `shape.max_context_tokens`, not the current
bucket, so one fixed allocation supports every graph variant. Return zero
workspace only for `single_block`. `auto_select` resolves through one internal
constant named `kDefaultOnlineDecodeMode`, initially `single_block` until Task
5's target gate decides promotion.

- [ ] **Step 4: Run focused and host tests**

Run:

```bash
RAFTINFER_RUN_GPU_TESTS=1 \
ctest --test-dir /home/charles/raftinfer-split-k-dev/build \
  -R '^raftinfer_qwen35_cuda_attention_test$' --output-on-failure
scripts/local-check.sh
git diff --check
```

Expected: the planning contract passes, existing online/materialized tests are
unchanged, and host checks pass.

- [ ] **Step 5: Commit**

```bash
git add cpp/execution/qwen35_execution_policy.hpp \
  cpp/kernels/qwen35_online_attention.cuh \
  cpp/kernels/qwen35_online_attention.cu \
  cpp/tests/qwen35_cuda_attention_test.cu
git commit -m "test: lock split-k decode planning contract"
```

---

### Task 2: Implement split-K partial scan and stable merge

**Files:**
- Modify: `cpp/kernels/qwen35_online_attention.cuh:19-35`
- Modify: `cpp/kernels/qwen35_online_attention.cu:307-420,697-782,861-919`
- Modify: `cpp/tests/qwen35_cuda_attention_test.cu:700-885`

**Interfaces:**
- Consumes: `Qwen35OnlineDecodePlan` and `Qwen35OnlineDecodeWorkspaceLayout` from Task 1.
- Produces the extended decode entry point:

```cpp
void qwen35_online_attention_decode(
    const void *query, const void *key, const void *value, const void *gate,
    void *output, void *kv_cache, std::size_t kv_cache_bytes,
    void *workspace, std::size_t workspace_bytes,
    Qwen35AttentionShape shape, RaftInferDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype, Qwen35KvCacheLayout cache_layout,
    Qwen35OnlineDecodePlan plan,
    const std::uint32_t *device_position, cudaStream_t stream);
```

The old single-block implementation remains the implementation selected by a
`single_block` plan and accepts `workspace == nullptr, workspace_bytes == 0`.

- [ ] **Step 1: Add failing boundary, tail, and partial-state tests**

Extend `run_online_decode_case` to accept a forced mode and explicit
`max_context_tokens`. Add BF16 activation/BF16 head-major cases for context
lengths:

```text
1, 255, 256, 257, 511, 512, 513, 777, 1024
```

For every case, allocate the exact workspace-layout byte count, fill it with a
sentinel, run decode, and compare final output with the materialized reference.
For `257`, `513`, and `777`, download the split workspace and independently
reconstruct each active partition's maximum, exponential sum, and value
accumulator on the host. Require the downloaded `partial_max`, `partial_sum`,
and `partial_value` entries to satisfy the existing BF16 tolerance.

Add one undersized-workspace assertion:

```cpp
expect_primitive_error([&] {
  raftinfer::kernels::qwen35_online_attention_decode(
      query, key, value, gate, output, cache, cache_bytes,
      workspace, required_workspace - 1, shape, RAFTINFER_DTYPE_BF16,
      raftinfer::Qwen35KvCacheDType::bf16,
      raftinfer::Qwen35KvCacheLayout::head_major, plan,
      device_position, context.stream());
});
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cmake --build /home/charles/raftinfer-split-k-dev/build \
  --target raftinfer_qwen35_cuda_attention_test
RAFTINFER_RUN_GPU_TESTS=1 \
ctest --test-dir /home/charles/raftinfer-split-k-dev/build \
  -R '^raftinfer_qwen35_cuda_attention_test$' --output-on-failure
```

Expected: the new boundary cases fail because the decode API and split-K
kernels are not implemented.

- [ ] **Step 3: Refactor the current scan recurrence into reusable device code**

Extract the per-warp recurrence from `online_decode_model_kernel` into a device
helper that accepts `[begin_token, end_token)` and returns lane-local output
plus warp-local max/sum. Preserve the exact score scale and FP32 recurrence:

```cpp
const float next_max = fmaxf(local_max, score);
const float left_scale =
    local_max == -CUDART_INF_F ? 0.0F : expf(local_max - next_max);
const float right_scale = expf(score - next_max);
local_sum = local_sum * left_scale + right_scale;
local_output[component] =
    local_output[component] * left_scale + value_component * right_scale;
local_max = next_max;
```

Use the helper from the existing single-block kernel first and rerun its
existing 32/128/512 tests before adding split-K behavior.

- [ ] **Step 4: Add the partition kernel**

Launch one block per `(query_head, active_partition_capacity)` for the plan's
fixed context bucket. Each block reads `device_position` when present, exits if
its partition starts beyond `position + 1`, scans only its partition, merges
its warps, and writes one SoA partial record. Specialize both 256-token and
512-token partition instantiations for activation/cache combinations already
supported by online decode.

Use a checked grid derived from:

```cpp
grid.x = shape.query_heads * plan.active_partition_capacity;
```

The KV append kernel remains ordered before the partition kernel on the same
RAFT stream.

- [ ] **Step 5: Add the per-head merge kernel**

Launch one block per query head. Merge only
`ceil((position + 1) / partition_tokens)` active records with:

```cpp
const float next_max = fmaxf(merged_max, partial_max);
const float left_scale = expf(merged_max - next_max);
const float right_scale = expf(partial_max - next_max);
merged_sum = merged_sum * left_scale + partial_sum * right_scale;
merged_value =
    merged_value * left_scale + partial_value * right_scale;
merged_max = next_max;
```

Apply the existing sigmoid gate and typed output store only after the final
division. Reject nonfinite/empty active-state conditions through validation,
not silent output substitution.

- [ ] **Step 6: Preserve short-context dispatch and validate every launch**

If `position + 1 < plan.split_k_threshold_tokens`, launch the unchanged
single-block kernel and leave the split workspace untouched. Otherwise launch
partition then merge. Every launch must retain named `cudaGetLastError`
checking. Unsupported signatures still fail before launch so the executor can
select its existing fallback during plan construction.

- [ ] **Step 7: Run all attention cases**

Run:

```bash
cmake --build /home/charles/raftinfer-split-k-dev/build \
  --target raftinfer_qwen35_cuda_attention_test
RAFTINFER_RUN_GPU_TESTS=1 \
ctest --test-dir /home/charles/raftinfer-split-k-dev/build \
  -R '^raftinfer_qwen35_cuda_attention_test$' --output-on-failure
git diff --check
```

Expected: all final outputs and downloaded partial states pass, including
non-multiple tails and both partition sizes.

- [ ] **Step 8: Commit**

```bash
git add cpp/kernels/qwen35_online_attention.cuh \
  cpp/kernels/qwen35_online_attention.cu \
  cpp/tests/qwen35_cuda_attention_test.cu
git commit -m "feat: add split-k online decode attention"
```

---

### Task 3: Integrate fixed scratch and context-bucket CUDA Graph variants

**Files:**
- Modify: `cpp/execution/qwen35_execution_policy.hpp:7-29`
- Modify: `cpp/execution/qwen35_executor.hpp:43-53`
- Modify: `cpp/execution/qwen35_executor.cu:275-292,448-574,635-791,918-1015,1191-1196,1913-1936,2050-2072`
- Modify: `cpp/execution/session.hpp:35-50`
- Modify: `cpp/execution/session.cpp:53-77`
- Modify: `cpp/tests/qwen35_executor_test.cu`
- Modify: `cpp/tests/qwen35_cuda_graph_test.cu:228-301`
- Modify: `cpp/tests/session_test.cpp`

**Interfaces:**
- Consumes: the Task 2 kernel entry point and fixed maximum workspace layout.
- Produces:

```cpp
namespace raftinfer {

enum class Qwen35DecodeAttentionImplementation : std::uint8_t {
  single_block,
  split_k,
};

struct Qwen35DecodeAttentionDiagnostic {
  Qwen35DecodeAttentionImplementation implementation{
      Qwen35DecodeAttentionImplementation::single_block};
  std::size_t partition_tokens{};
  std::size_t threshold_tokens{};
  std::size_t last_context_bucket_tokens{};
  bool split_k_graph_captured{};

  bool operator==(const Qwen35DecodeAttentionDiagnostic &) const = default;
};

} // namespace raftinfer
```

`Qwen35ExecutionDiagnostics` gains one
`Qwen35DecodeAttentionDiagnostic decode_attention` field. This is read-only
diagnostic state; request methods and sampling inputs do not change.

- [ ] **Step 1: Add failing executor workspace tests**

In `qwen35_executor_test.cu`, create the accepted 16/4/256 configuration with
`max_context=4096`, force `split-k-256`, and assert:

```cpp
policy.decode_attention =
    raftinfer::Qwen35DecodeAttentionMode::split_k_256;
const auto required = raftinfer::Qwen35Executor::workspace_bytes(
    config, 4096, policy);
raftinfer::Qwen35Executor executor{context, config, weights, 4096, policy};
const auto diagnostics = executor.diagnostics();
assert(diagnostics.attention_workspace_bytes ==
       raftinfer::kernels::qwen35_online_decode_workspace_layout(
           shape, raftinfer::Qwen35DecodeAttentionMode::split_k_256)
           .bytes);
assert(diagnostics.decode_attention.implementation ==
       raftinfer::Qwen35DecodeAttentionImplementation::split_k);
assert(diagnostics.decode_attention.partition_tokens == 256);
```

Also construct with exactly `required - 1` arena bytes and require the existing
workspace exhaustion error before any decode launch.

- [ ] **Step 2: Run executor tests and verify RED**

Run:

```bash
cmake --build /home/charles/raftinfer-split-k-dev/build \
  --target raftinfer_qwen35_executor_test
RAFTINFER_RUN_GPU_TESTS=1 \
ctest --test-dir /home/charles/raftinfer-split-k-dev/build \
  -R '^raftinfer_qwen35_executor_test$' --output-on-failure
```

Expected: compilation fails because policy resolution, scratch sizing, and
diagnostics are missing.

- [ ] **Step 3: Resolve the experimental mode once before workspace sizing**

Add `Qwen35DecodeAttentionMode decode_attention{auto_select}` to the internal
`Qwen35ExecutionPolicy`. Read `RAFTINFER_QWEN35_DECODE_ATTENTION` once in a
declared/testable `qwen35_execution_policy_from_environment` helper used by
`Session::Session`, and
accept exactly:

```text
auto
single-block
split-k-256
split-k-512
```

Unset means the caller's policy value, which defaults to `auto`. Empty or
unknown values throw `std::invalid_argument` during construction. Store the resolved
mode back into `policy_`, then pass that same policy object to both
`Qwen35Executor::workspace_bytes` and the `Qwen35Executor` constructor. Store
the maximum workspace layout as immutable executor plan data. Do not call
`getenv` from `Qwen35Executor`, `run_chunk`, `decode`, or graph replay.

In `session_test.cpp`, set each accepted value, call the helper, and assert the
exact enum. Test unset preservation, empty-string rejection, and unknown-value
rejection. Restore the original process environment after each case.

- [ ] **Step 4: Include split-K scratch in the one-time workspace estimate**

Make `max_attention_workspace` return the maximum of prefill/materialized
workspace and the fixed split-K decode layout. Allocate it once through the
existing `WorkspaceArena`. Pass the pointer, exact byte count, and selected
context-bucket plan into `qwen35_online_attention_decode`.

If the selected split-K mode is incompatible with the resolved model, dtype,
or cache layout, resolve to the existing compatible single-block/materialized
fallback before sizing. Diagnostics must report that fallback; forced mode
must never label an unsupported launch as split-K.

- [ ] **Step 5: Add graph variants keyed by decode context bucket**

Replace the single graph owner with a small vector of:

```cpp
struct DecodeGraphVariant {
  kernels::Qwen35OnlineDecodePlan attention_plan;
  std::unique_ptr<CudaGraphDecode> graph;
};
```

Provide exact helpers:

```cpp
DecodeGraphVariant &decode_graph_variant_for(std::size_t position);
void capture_decode_graph_variant(DecodeGraphVariant &variant);
```

Create variant descriptors for the single-block path plus only the split-K
buckets reachable for `max_context_`, then lazily capture each variant before
its first replay. Before capture, set an executor member
`captured_decode_attention_plan_`; `run_full_layer` consumes that immutable
value while recording the graph. Replay selects the variant from the known
host `position_` before launch.

Update `replay_decode_greedy` to select a variant on every loop iteration using
`start_position + step`. This allows a single coarse Rust FFI call to cross
the threshold or a 1024/2048/4096 bucket boundary without using the wrong grid.

- [ ] **Step 6: Add failing graph-boundary equivalence cases**

Extend `qwen35_cuda_graph_test.cu` with a `max_context=1030` executor. Prefill a
repeated valid-token prompt to positions 254, 255, 511, and 1023 in separate
reset runs, then decode across each next boundary. Compare token, logits, full
KV cache, convolution state, and recurrent state with graph-disabled execution.
Set `graph_policy.decode_attention` and `stream_policy.decode_attention` to
`Qwen35DecodeAttentionMode::split_k_256` before constructing both executors.

Require diagnostics after the 512 boundary:

```cpp
assert(diagnostics.decode_graph_replayed);
assert(diagnostics.decode_attention.implementation ==
       raftinfer::Qwen35DecodeAttentionImplementation::split_k);
assert(diagnostics.decode_attention.last_context_bucket_tokens == 1024);
assert(diagnostics.decode_attention.split_k_graph_captured);
```

- [ ] **Step 7: Run executor and graph tests**

Run:

```bash
cmake --build /home/charles/raftinfer-split-k-dev/build \
  --target raftinfer_qwen35_executor_test raftinfer_qwen35_cuda_graph_test
RAFTINFER_RUN_GPU_TESTS=1 \
ctest --test-dir /home/charles/raftinfer-split-k-dev/build \
  -R '^raftinfer_qwen35_(executor|cuda_graph)_test$' --output-on-failure
```

Expected: graph-on and graph-off observations are equal across all dispatch and
bucket boundaries, and no decode-time allocation occurs.

- [ ] **Step 8: Commit**

```bash
git add cpp/execution/qwen35_execution_policy.hpp \
  cpp/execution/qwen35_executor.hpp cpp/execution/qwen35_executor.cu \
  cpp/execution/session.hpp cpp/execution/session.cpp \
  cpp/tests/qwen35_executor_test.cu cpp/tests/qwen35_cuda_graph_test.cu \
  cpp/tests/session_test.cpp
git commit -m "feat: integrate split-k decode graph variants"
```

---

### Task 4: Propagate read-only diagnostics and add the promotion evaluator

**Files:**
- Modify: `cpp/execution/session.hpp:35-44`
- Modify: `cpp/execution/session.cpp:164-178`
- Modify: `cpp/include/raftinfer/c_api.h:35-74`
- Modify: `cpp/src/c_api.cpp:97-126,478-506`
- Modify: `cpp/tests/c_api_test.cpp`
- Modify: `rust/raftinfer-sys/src/lib.rs:74-85`
- Modify: `rust/raftinfer-runtime/src/lib.rs:202-211,620-637`
- Modify: `rust/raftinfer-runtime/tests/engine.rs`
- Modify: `rust/raftinfer-cli/src/main.rs:453-563,600-740`
- Modify: `scripts/qwen35-benchmark.sh:17-66,95-124,229-279`
- Create: `scripts/qwen35-split-k-gate.sh`
- Modify: `scripts/local-check.sh:1-18`
- Modify: `tests/benchmark-script-test.sh`
- Create: `tests/split-k-gate-script-test.sh`
- Modify: `docs/environment.md:22-46`

**Interfaces:**
- Consumes: `Qwen35DecodeAttentionDiagnostic` from Task 3.
- Produces additive C/Rust/JSON diagnostic fields:

```text
decode_attention: "single_block" | "split_k"
decode_attention_partition_tokens: integer >= 0
decode_attention_threshold_tokens: integer >= 0
decode_attention_context_bucket_tokens: integer >= 0
decode_attention_split_k_graph_captured: boolean
```

- Produces:

```bash
scripts/qwen35-split-k-gate.sh BASELINE_JSONL CANDIDATE_A_JSONL CANDIDATE_B_JSONL
```

- [ ] **Step 1: Write failing ABI and JSON tests**

Add the diagnostic enum constants and struct-field expectations to
`c_api_test.cpp`, `rust/raftinfer-runtime/tests/engine.rs`, and the CLI JSON unit
tests. Require the exact JSON fragment:

```json
{"decode_attention":"split_k","decode_attention_partition_tokens":256,"decode_attention_threshold_tokens":256,"decode_attention_context_bucket_tokens":1024,"decode_attention_split_k_graph_captured":true}
```

Keep `schema_version` at 2 because the fields are additive diagnostics and the
existing required fields retain their meaning.

- [ ] **Step 2: Run host tests and verify RED**

Run:

```bash
cargo test --workspace
scripts/local-check.sh
```

Expected: C/Rust diagnostics and CLI JSON tests fail because the fields are
missing.

- [ ] **Step 3: Propagate diagnostics without adding request calls**

Add `RAFTINFER_QWEN35_DECODE_ATTENTION_SINGLE_BLOCK = 0` and
`RAFTINFER_QWEN35_DECODE_ATTENTION_SPLIT_K = 1`. Append the five fields to
`RaftInferSessionDiagnostics`, mirror their order in `raftinfer-sys`, map them
through `SessionDiagnostics` and `ExecutionDiagnostics`, and serialize them in
both generation and benchmark JSON.

No new FFI function is added. Rust still obtains all fields through the one
existing `raftinfer_session_diagnostics` call.

- [ ] **Step 4: Make benchmark evidence reject undisclosed paths**

Extend `qwen35-benchmark.sh` validation so every parity and benchmark execution
object contains all five fields. When
`RAFTINFER_QWEN35_DECODE_ATTENTION=split-k-256` or `split-k-512`, require the
PP512 and TG128@PP512 records to report `decode_attention == "split_k"`, the
matching partition size, a positive context bucket, and a captured split-K
graph. PP128 may report the single-block short-context path.

Document the environment variable and its four exact values in
`docs/environment.md`.

Append `tests/split-k-gate-script-test.sh` to `scripts/local-check.sh` so the
promotion evaluator is part of every host verification run.

- [ ] **Step 5: Write the failing split-K promotion-gate fixture**

Create `tests/split-k-gate-script-test.sh` with one baseline and two candidate
three-record JSONL fixtures. The passing fixture must satisfy:

```text
candidate PP512 generation >= baseline * 1.01
candidate TG128@PP512 generation >= baseline * 1.01
candidate PP128 prefill >= baseline * 0.99
candidate PP512 prefill >= baseline * 0.99
candidate PP128 generation >= baseline * 0.99
candidate exact parity = 128/128
candidate RAFTInfer CV <= 0.03
candidate long-context decode_attention = split_k
```

Add independent failing mutations for each metric, one missing arm, one
artifact mismatch, one GPU mismatch, one nonfinite value, one unstable CV, one
single candidate file, and one undisclosed/fallback decode path.

- [ ] **Step 6: Implement the deterministic gate**

`qwen35-split-k-gate.sh` must validate exactly three records in every file,
join records by arm rather than line position, and require matching model SHA,
GPU name, driver, CUDA, RAFT, RMM, warmup count, and measured count. It must
print:

```text
qwen35-split-k-gate: pass baseline=... candidate_a=... candidate_b=...
```

only after both candidate files independently pass every threshold.

- [ ] **Step 7: Run all host/script tests**

Run:

```bash
scripts/local-check.sh
git diff --check
```

Expected: all host, ABI, Rust, JSON, benchmark-harness, and split-K gate tests
pass.

- [ ] **Step 8: Commit**

```bash
git add cpp/execution/session.hpp cpp/execution/session.cpp \
  cpp/include/raftinfer/c_api.h cpp/src/c_api.cpp cpp/tests/c_api_test.cpp \
  rust/raftinfer-sys/src/lib.rs rust/raftinfer-runtime/src/lib.rs \
  rust/raftinfer-runtime/tests/engine.rs rust/raftinfer-cli/src/main.rs \
  scripts/qwen35-benchmark.sh scripts/qwen35-split-k-gate.sh \
  scripts/local-check.sh \
  tests/benchmark-script-test.sh tests/split-k-gate-script-test.sh \
  docs/environment.md
git commit -m "feat: disclose and gate split-k decode attention"
```

---

### Task 5: Tune on RTX 5090 and promote only a passing default

**Files:**
- Modify after a passing target decision: `cpp/kernels/qwen35_online_attention.cu`
- Modify: `cpp/tests/qwen35_cuda_attention_test.cu`
- Modify when formal evidence changes: `benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl`
- Regenerate when formal evidence changes: `docs/assets/qwen35-bf16-rtx5090.svg`
- Modify: `docs/benchmarks.md`
- Modify: `docs/verification/raftinfer-release.md`
- Modify when displayed values change: `README.md`, `README.zh-CN.md`

**Interfaces:**
- Consumes: a clean candidate commit, target `charles@192.168.124.8`, the pinned
  BF16 model/provenance, pinned llama.cpp server, and Tasks 1-4 gates.
- Produces: a deterministic `auto` decision of `single_block`, `split_k_256`,
  or `split_k_512`, plus retained test/parity/benchmark evidence.

- [ ] **Step 1: Run local preflight before target transfer**

Run:

```bash
scripts/local-check.sh
git diff --check
git status --short
```

Expected: all host checks pass and only intentional implementation changes are
present before their task commits; begin target validation from a clean commit.

- [ ] **Step 2: Sync into a commit-specific target directory**

Set safe target variables and use the checked-in synchronizer:

```bash
export RAFTINFER_TARGET='charles@192.168.124.8'
export RAFTINFER_VALIDATION_ROOT='/home/charles/raftinfer-split-k-validation'
export RAFTINFER_CANDIDATE_SHORT="$(git rev-parse --short=12 HEAD)"
export RAFTINFER_TARGET_DIR="${RAFTINFER_VALIDATION_ROOT}/source-${RAFTINFER_CANDIDATE_SHORT}"
scripts/sync-target.sh
```

Record the local commit, source archive hash, and remote source hash. Do not
transfer `.git`, build directories, credentials, or untracked evidence.

- [ ] **Step 3: Verify the shared GPU and build a clean Release candidate**

On the target, require `scripts/gpu-preflight.sh` to pass and inspect
`nvidia-smi` compute applications before each measurement. Do not stop an
unrelated process. Configure:

```bash
export source_root="${RAFTINFER_TARGET_DIR}"
export build_root="${RAFTINFER_VALIDATION_ROOT}/build-${RAFTINFER_CANDIDATE_SHORT}"
cmake -S "${source_root}" -B "${build_root}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAFTINFER_ENABLE_CUDA=ON \
  -DRAFTINFER_BUILD_TESTS=ON \
  -DRAFTINFER_NATIVE_LIBRARY_TYPE=SHARED
cmake --build "${build_root}" --parallel
```

- [ ] **Step 4: Run the complete CUDA test suite**

Run under the cooperative lock:

```bash
RAFTINFER_RUN_GPU_TESTS=1 \
LD_LIBRARY_PATH="${build_root}/cpp:${RAFTINFER_RUNTIME_LIB}" \
ctest --test-dir "${build_root}" --output-on-failure
```

Expected: all required tests execute and pass with no skip/not-run marker.

- [ ] **Step 5: Establish a same-window single-block baseline**

Run exact parity and `scripts/qwen35-benchmark.sh` with:

```bash
export RAFTINFER_QWEN35_DECODE_ATTENTION='single-block'
export PARITY_OUTPUT="${RAFTINFER_VALIDATION_ROOT}/single-block-parity.jsonl"
export BENCHMARK_OUTPUT="${RAFTINFER_VALIDATION_ROOT}/single-block-benchmark.jsonl"
```

Require 128/128 exact greedy tokens, BF16 head-major cache, graph replay, 5
warmups, 20 measured iterations, CV at most 3%, and the existing BF16 llama.cpp
gate. Retain SHA-256 hashes for parity and benchmark outputs.

- [ ] **Step 6: Run one exploratory measurement for each split-K candidate**

Repeat parity and benchmark once with each exact value:

```text
RAFTINFER_QWEN35_DECODE_ATTENTION=split-k-256
RAFTINFER_QWEN35_DECODE_ATTENTION=split-k-512
```

Discard any run in which preflight fails, another compute process appears, a
required diagnostic reports fallback, parity is not 128/128, or CV exceeds 3%.

Choose the candidate with the larger value of:

```text
min(
  candidate_pp512_generation_tps / baseline_pp512_generation_tps,
  candidate_tg128_pp512_generation_tps / baseline_tg128_pp512_generation_tps
)
```

If the values are exactly equal, choose `split-k-512` because it uses less
scratch and fewer partial records.

- [ ] **Step 7: Run two independent formal samples for the selected candidate**

Run the selected mode twice with fresh GPU preflight and distinct output files.
Then run:

```bash
scripts/qwen35-split-k-gate.sh \
  "${RAFTINFER_VALIDATION_ROOT}/single-block-benchmark.jsonl" \
  "${RAFTINFER_VALIDATION_ROOT}/split-k-candidate-a.jsonl" \
  "${RAFTINFER_VALIDATION_ROOT}/split-k-candidate-b.jsonl"
```

Expected: exit 0 only if both formal samples independently satisfy every
correctness, stability, disclosure, long-context gain, and non-regression gate.

- [ ] **Step 8: Apply the deterministic default decision**

If Step 7 passes, change `kDefaultOnlineDecodeMode` from `single_block` to the
selected `split_k_256` or `split_k_512` value and add a static/default-plan test
for that exact choice. If Step 7 fails, leave the constant at `single_block`
and add a test proving forced split-K remains selectable while `auto` resolves
to the safe default.

No threshold is weakened and no favorable single run is substituted for the
two-run gate.

- [ ] **Step 9: Rebuild and rerun final default-path verification**

After changing the default constant, rebuild from a fresh commit-specific
directory and rerun:

```text
complete CUDA CTest
128/128 exact greedy parity with RAFTINFER_QWEN35_DECODE_ATTENTION=auto
two independent qwen35-benchmark.sh runs
qwen35-bf16-gate.sh on both runs
qwen35-split-k-gate.sh against the same-window single-block baseline
```

Expected after promotion: diagnostics disclose split-K on both long-context
arms and all gates pass. Expected without promotion: correctness passes,
diagnostics disclose single-block for `auto`, and the failed performance gate
is documented rather than hidden.

- [ ] **Step 10: Refresh evidence and documentation**

Update `docs/verification/raftinfer-release.md` with candidate commit, build and
binary hashes, test count, parity hash, baseline hash, both candidate hashes,
resolved partition size, context threshold/buckets, per-arm medians/CVs, and
the promotion decision.

Only after a passing default promotion, replace the checked-in conservative
benchmark sample, regenerate the SVG, and update README/doc values. Continue to
compare RAFTInfer with llama.cpp only; do not add a bw24 chart series.

- [ ] **Step 11: Run publication checks**

Run:

```bash
scripts/local-check.sh
scripts/qwen35-bf16-gate.sh benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl
python3 tests/benchmark-chart-test.py
tests/benchmark-asset-test.sh
git diff --check
```

Expected: all checks pass and documentation points only to retained,
hash-identified evidence.

- [ ] **Step 12: Commit the measured decision**

```bash
git add cpp/kernels/qwen35_online_attention.cu \
  cpp/tests/qwen35_cuda_attention_test.cu \
  benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl \
  docs/assets/qwen35-bf16-rtx5090.svg docs/benchmarks.md \
  docs/verification/raftinfer-release.md README.md README.zh-CN.md
git commit -m "perf: select measured split-k decode default"
```

If split-K is not promoted, stage only the default-selection test and the
verification/benchmark documentation that records the failed gate, and use:

```bash
git commit -m "perf: retain single-block decode default"
```

---

### Task 6: Final review and phase-B handoff

**Files:**
- Modify only for review findings: files changed by Tasks 1-5
- Create after phase A is accepted: `docs/superpowers/specs/2026-08-02-decode-gemv-gated-delta-design.md`

**Interfaces:**
- Consumes: the phase-A implementation and retained target evidence.
- Produces: a review-clean phase-A branch and a separately approved phase-B
  design boundary; phase B implementation is not mixed into phase A commits.

- [x] **Step 1: Review the complete phase-A diff**

Inspect:

```bash
git diff 810215b...HEAD --stat
git diff 810215b...HEAD -- cpp/kernels cpp/execution cpp/include cpp/src \
  rust scripts tests docs benchmarks README.md README.zh-CN.md
git log --oneline 810215b..HEAD
```

Check specifically for device-position boundary errors, graph variant/address
stability, workspace overflow, hidden fallback, environment parsing on the hot
path, ABI field-order mismatch, stale benchmark schemas, and accidental bw24
comparison claims.

- [x] **Step 2: Run final verification from clean state**

Require a clean worktree, rerun `scripts/local-check.sh`, and rerun the clean
target CUDA build/test/parity/benchmark commands from Task 5. Verify every
reported SHA-256 against the retained file.

- [x] **Step 3: Commit review fixes if required**

If review changes files:

```bash
git add cpp rust scripts tests docs benchmarks README.md README.zh-CN.md
git commit -m "fix: complete split-k decode review"
```

Skip this commit only when the review produces no changes.

- [ ] **Step 4: Start phase B with a separate approved design**

Create the phase-B design around decode GEMV/small-M paths, grouped/fused
projection and cast removal, fused RMSNorm where numerically safe, and complete
Gated DeltaNet launch/traffic fusion. Preserve the same layered correctness,
exact-token, target-GPU, fallback, provenance, and measured-promotion rules.
Do not implement phase B until that design is reviewed and approved.
