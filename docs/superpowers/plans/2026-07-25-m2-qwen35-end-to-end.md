# M2 Qwen3.5-9B End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the text path of a converted Qwen3.5-9B BF16/F16 GGUF end to end
on RTX 50, with exact greedy-token parity against a pinned llama.cpp build.

**Architecture:** C++ owns the mapped GGUF, immutable GPU weights, hybrid
Gated-DeltaNet/full-attention session state, and complete prefill/decode steps.
Independent CPU FP32 code defines operator and state-transition semantics.
Rust parses the versioned tokenizer specification, applies the text chat
template, and calls C++ once per prefill or decode step.

**Tech Stack:** C++20, CUDA 13, cuBLASLt, RAFT 26.06, RMM 26.06, CMake/CTest,
Rust 2024, GGUF v3, pinned llama.cpp.

## Global Constraints

- Target only NVIDIA consumer Blackwell RTX 50 (`sm_120a`); RTX 40 is out of
  scope.
- Model target is the text-generation path of `Qwen/Qwen3.5-9B`: 32 dense-MLP
  blocks, 24 Gated DeltaNet blocks, and 8 full-attention blocks in an exact
  3-linear/1-full order.
- M2 accepts F16 or BF16 primary GGUF weights and preserves required FP32
  auxiliary tensors. Q4_K_M, vision, mmproj, MTP, speculative decoding, CUDA
  Graphs, and multi-session scheduling remain out of scope.
- RAFT/RMM remain the resource and memory foundation. C++ owns CUDA resources
  and GPU pointers; Rust uses only coarse model/session calls.
- Decode performs no device allocation. Session reset clears logical state
  without reallocating.
- GGUF RMSNorm tensors are direct learned scales. Full attention uses Q/K
  normalization, partial RoPE, and a sigmoid output gate; linear attention
  retains convolution state and FP32 recurrent state. GGUF `ssm_a` is the
  direct negative recurrent coefficient used by `exp(ssm_a * dt)`.
- Functional/algorithmic correctness and kernel performance are both gates.
  An optimized or imported kernel is selected only when it passes the same
  numerical test as its baseline and is at least 1.05x faster for its declared
  shape bucket.
- The inspected `bw24` sources contain no reusable Qwen3.5 kernel. New native
  kernels must therefore be provenance-labeled `project-native`; if a real
  `bw24` kernel is later supplied, direct reuse requires license, provenance,
  correctness, and RTX 50 benchmark evidence.
- GPU work runs only after `scripts/gpu-preflight.sh` reports no compute
  processes, at least 2048 MiB free, and at most 5% utilization.
- Operator checks use `abs <= 2e-2` or `rel <= 2e-2` for CUDA against CPU FP32.
  End-to-end greedy token IDs must match llama.cpp exactly at every generated
  step.

---

### Task 0: Close M2A contract gaps

**Files:**

- Modify: `cpp/model/gguf_types.hpp`
- Modify: `cpp/model/gguf_reader.cpp`
- Modify: `cpp/model/qwen35_config.cpp`
- Modify: `cpp/model/model.hpp`
- Modify: `cpp/model/model.cpp`
- Modify: `cpp/tests/gguf_reader_test.cpp`
- Modify: `cpp/tests/qwen35_config_test.cpp`

**Interfaces:**

- Preserves: `gguf::read_catalog`, `derive_qwen35_config`, and `Model`.
- Produces: bounded catalog-memory accounting, authoritative
  `qwen35.layer_types` handling when present, and checked mapped tensor
  payload spans for Task 5.

- [x] **Step 1: Add failing contract tests**

  Require a present 32-entry `qwen35.layer_types` string array to exactly match
  the accepted `linear_attention`/`full_attention` vocabulary and order.
  Reject an array that disagrees with `full_attention_interval`, a catalog
  whose cumulative key/name/dimension/value storage exceeds its configured
  byte limit, and a payload request outside the mapped file.

- [x] **Step 2: Observe RED**

  Run the GGUF and Qwen3.5 config tests and confirm each new case fails for the
  missing behavior.

- [x] **Step 3: Implement bounded accounting and ordered-layer parsing**

  Add `ReaderLimits::max_catalog_bytes`, charge every retained key, string,
  array element, tensor name, and dimension before allocation, and reject on
  checked overflow. Use the layer array as authoritative and the interval only
  as a consistency check; retain interval-derived order only when the array is
  absent.

- [x] **Step 4: Add a checked model payload view**

  Expose a const span derived from
  `tensor_data_offset + TensorInfo::offset + byte_size` while the mapping
  remains owned by `Model`. Never expose the mapping base or a mutable pointer.

- [x] **Step 5: Run tests and commit**

  Commit: `fix: close Qwen3.5 M2A validation gaps`

### Task 1: Qwen3.5 FP32 reference primitives

**Files:**

- Create: `cpp/reference/qwen35.hpp`
- Create: `cpp/reference/qwen35.cpp`
- Create: `cpp/tests/qwen35_reference_test.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: `raftinfer::model::Qwen35Config`.
- Produces:
  - `qwen35_rms_norm(input, weight, output, rows, cols, epsilon)`
  - `qwen35_gated_full_attention(input, output, FullAttentionReferenceArgs)`
  - `qwen35_gated_delta_step(input, output, GatedDeltaReferenceArgs, state)`
  - `qwen35_gated_delta_prefill(input, output, GatedDeltaReferenceArgs, state)`

- [x] **Step 1: Write failing RMSNorm and gated-attention tests**

  Use hand-computed two-row RMSNorm values to prove the multiplier is
  `1 + weight`, not `weight`. Use a two-token/two-head attention fixture where
  Q/K normalization, the causal mask, partial RoPE, and `sigmoid(z)` each
  change the expected output.

- [x] **Step 2: Run the reference target and observe RED**

  Run:
  `cmake --build build/host --target raftinfer_qwen35_reference_test -j4`

  Expected: compilation fails because `reference/qwen35.hpp` and its symbols
  do not exist.

- [x] **Step 3: Implement checked FP32 primitives**

  Validate every span and dimension before calculation. Accumulate norm,
  attention dot products, and recurrent-state outer products in FP32 or
  double where the reference formula requires it. Implement causal prefill as
  repeated state transitions so continued prefill and decode share one
  semantic path.

- [x] **Step 4: Add Gated DeltaNet transition vectors**

  Cover convolution shift/update, `softplus(dt_bias)`, decay
  `exp(-exp(A) * dt)`, beta interpolation, normalized key/query, recurrent
  matrix update, and gated normalized output. Compare one prefill call with
  repeated single-token calls.

- [x] **Step 5: Run tests and commit**

  Run:
  `ctest --test-dir build/host -R raftinfer_qwen35_reference_test --output-on-failure`

  Commit: `feat: add Qwen3.5 hybrid FP32 references`

### Task 2: Hybrid session-state contract

**Files:**

- Create: `cpp/execution/qwen35_state.hpp`
- Create: `cpp/execution/qwen35_state.cpp`
- Create: `cpp/tests/qwen35_state_test.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: `Qwen35Config::blocks`.
- Produces:
  - `Qwen35StateLayout::create(config, max_context_tokens)`
  - `Qwen35HostState(layout)`
  - `position()`, `full_kv_length(layer)`, `linear_convolution(layer)`,
    `linear_recurrent(layer)`, `commit_tokens(count)`, and `reset()`

- [x] **Step 1: Write failing layout tests**

  Assert a four-block fixture creates three convolution/recurrent slots and
  one K/V slot, while the official 32-block configuration creates 24 and 8.
  Checked arithmetic must reject zero context and byte-count overflow.

- [x] **Step 2: Verify RED**

  Run:
  `cmake --build build/host --target raftinfer_qwen35_state_test -j4`

  Expected: missing state-layout symbols.

- [x] **Step 3: Implement immutable layout and preallocated host state**

  Store convolution as
  `[linear_layer][qkv_channel][conv_width - 1]`, recurrent state as
  `[linear_layer][value_head][key_dim][value_dim]` in FP32, and K/V as
  `[full_layer][K_or_V][token][kv_head][head_dim]`.

- [x] **Step 4: Add atomic logical commit and reset tests**

  A failed bounds check must leave position and every layer length unchanged.
  `reset()` must zero all state and retain vector capacities and base
  addresses.

- [x] **Step 5: Run tests and commit**

  Commit: `feat: define Qwen3.5 hybrid session state`

### Task 3: Official layer fixtures and reference conformance

**Files:**

- Create: `tools/qwen35/export_reference_fixtures.py`
- Create: `tests/fixtures/qwen35/README.md`
- Create: `tests/fixtures/qwen35/tiny-layer-v1.bin`
- Create: `cpp/tests/qwen35_fixture_test.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: the pinned official Transformers Qwen3.5 implementation.
- Produces: a versioned binary fixture containing config, inputs, weights,
  initial states, full-attention outputs, linear-attention outputs, and final
  states.

- [x] **Step 1: Write a fixture reader test before adding the fixture**

  The test must reject bad magic/version/length and compare every stored
  output/state field with the Task 1 reference implementation at
  `abs <= 1e-5` or `rel <= 1e-5`.

- [x] **Step 2: Observe RED because the fixture is absent**

- [x] **Step 3: Export a deterministic small fixture**

  Use fixed seed `20260725`, CPU float32, batch 1, sequence length 4, hidden
  size 8, two full-attention heads, one KV head, and two linear value heads.
  Record the exact Transformers commit in the fixture README.

- [x] **Step 4: Run conformance tests and commit**

  Commit: `test: pin Qwen3.5 hybrid layer semantics`

### Task 4: Stable session ABI and host ownership

**Files:**

- Modify: `cpp/include/raftinfer/c_api.h`
- Modify: `cpp/src/c_api.cpp`
- Create: `cpp/execution/session.hpp`
- Create: `cpp/execution/session.cpp`
- Modify: `cpp/model/model.hpp`
- Modify: `cpp/model/model.cpp`
- Modify: `cpp/tests/c_api_test.cpp`
- Modify: `rust/raftinfer-sys/src/lib.rs`
- Modify: `rust/raftinfer-runtime/src/lib.rs`
- Modify: `rust/raftinfer-runtime/tests/engine.rs`

**Interfaces:**

- Produces:

  ```c
  typedef struct RaftInferSessionHandle RaftInferSessionHandle;
  typedef struct RaftInferSessionConfig {
    size_t struct_size;
    uint32_t max_context_tokens;
  } RaftInferSessionConfig;
  typedef struct RaftInferTokenResult {
    int32_t token_id;
    uint32_t position;
  } RaftInferTokenResult;
  RaftInferStatus raftinfer_session_create(
      RaftInferModelHandle*, const RaftInferSessionConfig*, RaftInferSessionHandle**);
  RaftInferStatus raftinfer_session_prefill(
      RaftInferSessionHandle*, const int32_t*, size_t, RaftInferTokenResult*);
  RaftInferStatus raftinfer_session_decode(
      RaftInferSessionHandle*, int32_t, RaftInferTokenResult*);
  RaftInferStatus raftinfer_session_reset(RaftInferSessionHandle*);
  void raftinfer_session_destroy(RaftInferSessionHandle*);
  ```

- [x] **Step 1: Add C and Rust compile/lifetime tests**

  Prove null/size/context validation, atomic creation failure, model lifetime
  ownership, reset behavior, and that Rust cannot outlive the model through
  `Session<'model>`.

- [x] **Step 2: Observe RED for missing ABI**

- [x] **Step 3: Implement opaque ownership**

  `RaftInferSessionHandle` owns one `raftinfer::Session`. Rust `Session<'model>` owns only
  the native handle and carries `PhantomData<&'model Model<'engine>>`.
  Prefill/decode return `RAFTINFER_STATUS_UNAVAILABLE` in host-only builds.

- [x] **Step 4: Run host CTest/Cargo tests and commit**

  Commit: `feat: add coarse Qwen3.5 session ABI`

### Task 5: Immutable CUDA weight plan

**Files:**

- Create: `cpp/model/cuda_weights.hpp`
- Create: `cpp/model/cuda_weights.cu`
- Modify: `cpp/model/model.cpp`
- Modify: `cpp/foundation/device_context.hpp`
- Modify: `cpp/foundation/device_context.cu`
- Create: `cpp/tests/cuda_weights_test.cu`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: mapped tensor payload addresses from `Model`, semantic bindings
  from `Qwen35Manifest`, and the RMM device resource.
- Produces: `CudaWeightPlan` with fixed device addresses and typed views for
  embeddings, every block tensor, final norm, and LM head.

- [x] **Step 1: Add CUDA tests for F16/BF16 copy and unsupported type**

  Use the synthetic four-block GGUF. Verify bytes copied to fixed device
  allocations and an F32 or quantized primary tensor is rejected for M2.

- [x] **Step 2: Observe RED on the target container**

- [x] **Step 3: Expose tensor payload spans from the mapped model**

  A payload address is
  `mapped_base + catalog.tensor_data_offset + TensorInfo::offset`; validate the
  full span again before upload.

- [x] **Step 4: Allocate/upload atomically with RMM**

  Allocate all buffers before publishing `CudaWeightPlan`; synchronize the
  upload stream before success. Destruction uses the same device resource.

- [x] **Step 5: Run GPU tests and commit**

  Commit: `feat: upload immutable Qwen3.5 weights`

### Task 6: CUDA primitive and state-update kernels

**Files:**

- Create: `cpp/kernels/qwen35_primitives.cuh`
- Create: `cpp/kernels/qwen35_primitives.cu`
- Create: `cpp/kernels/qwen35_attention.cu`
- Create: `cpp/kernels/qwen35_delta.cu`
- Create: `cpp/tests/qwen35_cuda_kernels_test.cu`
- Modify: `cpp/registry/operator_registry.hpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**

- Produces project-native kernels for embedding lookup, direct-scale RMSNorm,
  residual add, partial RoPE, Q/K normalization, causal attention,
  sigmoid-gated attention output, SwiGLU, convolution shift/update, recurrent
  delta update, output normalization/gating, and deterministic argmax.

- [x] **Step 1: Add parameterized BF16/F16 tests against Task 1**

  Cover prefill lengths 1, 2, 4, and 17; decode after prefill; continued
  prefill; state reset; tail dimensions; and invalid shapes. Require
  `abs <= 2e-2` or `rel <= 2e-2`.

- [x] **Step 2: Observe RED on RTX 50**

- [x] **Step 3: Implement correctness-first kernels**

  Every launch uses the session stream and preallocated workspace. No kernel
  creates streams or allocates memory. Recurrent accumulation remains FP32.

- [x] **Step 4: Register complete capabilities**

  Include dtype, rank, alignment, shape bucket, `sm_120a`, deterministic flag,
  workspace size, and provenance `project-native`.

- [x] **Step 5: Benchmark before selection**

  Record median of 100 warm runs after 20 warmups. A fused alternative replaces
  the baseline only when it passes numerical tests and reaches at least 1.05x
  speedup for that bucket.

- [x] **Step 6: Run tests and commit**

  Commit: `feat: add Qwen3.5 CUDA state kernels`

### Task 7: cuBLASLt projections and complete GPU executor

**Files:**

- Create: `cpp/execution/cublaslt_matmul.hpp`
- Create: `cpp/execution/cublaslt_matmul.cu`
- Create: `cpp/execution/qwen35_executor.hpp`
- Create: `cpp/execution/qwen35_executor.cu`
- Modify: `cpp/execution/session.cpp`
- Modify: `cpp/foundation/device_context.hpp`
- Modify: `cpp/foundation/device_context.cu`
- Create: `cpp/tests/qwen35_executor_test.cu`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**

- Produces:
  - `Qwen35Executor::prefill(tokens, state, result)`
  - `Qwen35Executor::decode(token, state, result)`
  - fixed cuBLASLt algorithms/workspaces for embedding-to-logits execution.

- [x] **Step 1: Add a tiny four-block end-to-end GPU test**

  Compare every block output, state transition, final logits, and argmax with
  a CPU FP32 execution assembled from Task 1.

- [x] **Step 2: Observe RED**

- [x] **Step 3: Implement projection plans**

  Create cuBLASLt descriptors/heuristics during model/session initialization,
  never during decode. Support GGUF F16 and BF16 input weights with FP32
  accumulation and activation output in the configured 16-bit type.

- [x] **Step 4: Implement ordered hybrid execution**

  Preserve exact block order. Compute attention/DeltaNet, residual, MLP, final
  norm, LM-head logits, and deterministic smallest-index argmax. Commit all
  layer state lengths only after every launch succeeds.

- [x] **Step 5: Prove no decode allocation**

  Install a counting RMM resource around one session, warm it, run 32 decode
  steps, and assert allocation/deallocation counters do not change.

- [x] **Step 6: Run tests and commit**

  Commit: `feat: execute Qwen3.5 prefill and decode on CUDA`

### Task 8: Rust tokenizer-spec parser, BPE, and chat template

**Files:**

- Create: `rust/raftinfer-runtime/src/tokenizer.rs`
- Modify: `rust/raftinfer-runtime/src/lib.rs`
- Create: `rust/raftinfer-runtime/tests/tokenizer.rs`
- Create: `tests/parity/qwen35-tokenizer-corpus.jsonl`

**Interfaces:**

- Consumes: tokenizer-spec version 1 from `Model::tokenizer_spec()`.
- Produces:
  - `Tokenizer::from_spec(&TokenizerSpec)`
  - `Tokenizer::encode(text, add_special_tokens)`
  - `Tokenizer::decode(tokens, skip_special_tokens)`
  - `Tokenizer::apply_chat_template(messages, add_generation_prompt)`

- [x] **Step 1: Add binary-format corruption tests**

  Reject bad magic/version, truncation at every field category, duplicate
  keys, wrong scalar/array types, missing tokens/merges/model/pre fields, and
  out-of-range special IDs.

- [x] **Step 2: Observe RED**

- [x] **Step 3: Parse the deterministic metadata map**

  Preserve arbitrary byte tokens, merges, token types, BOS/EOS/EOT/padding IDs,
  pre-tokenizer identifier, add-BOS/add-EOS flags, and chat template.

- [x] **Step 4: Implement Qwen byte-level BPE without a second GGUF reader**

  Split contractions, Unicode letters, numbers, punctuation, and whitespace;
  protect registered special tokens; merge by GGUF rank; and use byte fallback
  for unmatched UTF-8. Decode reverses byte encoding before UTF-8 replacement.

- [x] **Step 5: Add pinned llama.cpp token vectors**

  Cover English, Simplified Chinese, mixed punctuation, whitespace/newlines,
  emoji, special-token text, and the no-tools Qwen3.5 chat template.

- [x] **Step 6: Run Cargo tests and commit**

  Commit: `feat: tokenize Qwen3.5 prompts in Rust`

### Task 9: Rust generation API and CLI

**Files:**

- Modify: `rust/raftinfer-runtime/src/lib.rs`
- Modify: `rust/raftinfer-cli/src/main.rs`
- Modify: `rust/raftinfer-cli/tests/cli.rs`
- Create: `tests/parity/qwen35-generation-corpus.jsonl`

**Interfaces:**

- Produces `Model::create_session`, `Session::prefill`, `Session::decode`,
  `Session::reset`, and:

  ```text
  raftinfer-cli generate --model MODEL.gguf --prompt TEXT
                   --max-new-tokens N --context N
  ```

- [x] **Step 1: Add CLI argument/error tests**

  Require model, prompt, positive context, and positive max-new-tokens. Reject
  context overflow and host-only execution with actionable errors.

- [x] **Step 2: Observe RED**

- [x] **Step 3: Implement deterministic greedy generation**

  Tokenize once, submit one prefill call, then one decode call per generated
  token until EOS/EOT or the limit. Do not expose per-layer/operator calls.

- [x] **Step 4: Add reset/reuse integration test and commit**

  Commit: `feat: add Qwen3.5 greedy generation CLI`

### Task 10: Real artifact, llama.cpp parity, and performance evidence

**Files:**

- Create: `scripts/prepare-qwen35-gguf.sh`
- Create: `scripts/qwen35-parity.sh`
- Create: `scripts/qwen35-benchmark.sh`
- Create: `docs/provenance/qwen35-9b.md`
- Create: `docs/verification/m2.md`
- Modify: `README.md`

**Interfaces:**

- Consumes: official Qwen3.5-9B checkpoint and a pinned llama.cpp checkout.
- Produces: a local BF16/F16 GGUF, SHA-256/provenance record, fixed-corpus
  token vectors, parity report, and prefill/decode performance record.

- [x] **Step 1: Pin artifact inputs**

  Record official model revision, Transformers revision, llama.cpp converter
  revision, llama.cpp reference revision, exact conversion command, output
  GGUF size, and SHA-256. Keep model files out of Git.

- [x] **Step 2: Establish llama.cpp golden outputs**

  Use temperature 0, batch 1, context 4096, and 32 generated tokens for at
  least four prompts: English factual, English code, Simplified Chinese, and
  mixed Chinese/English punctuation. Store token IDs, not only decoded text.

- [x] **Step 3: Run RAFTINFER parity under GPU preflight**

  Require prompt tokens and every greedy generated token to match exactly.
  On the first mismatch, record step, expected/actual token, and the nearest
  available block/logit diagnostic.

- [x] **Step 4: Benchmark only after parity**

  Measure prompt processing at 128 and 512 tokens and decode for 128 tokens,
  with 5 warmups and 20 measured runs. Record median, p95, tokens/s, peak
  allocated GPU bytes, GPU model/driver, CUDA/RAFT/RMM versions, clocks, and
  llama.cpp comparison. Flag any RAFTINFER result below 0.8x llama.cpp for follow-up
  before M2 is accepted.

- [x] **Step 5: Run all host and GPU checks**

  Run fresh Debug and Release host builds, `scripts/local-check.sh`, target
  CUDA build/CTest, tokenizer corpus, end-to-end parity, and benchmark scripts.

- [x] **Step 6: Update milestone documentation and commit**

  Mark M2 complete only when all correctness gates pass. Record performance
  honestly even when it exceeds the 0.8x floor.

  Commit: `docs: record Qwen3.5 M2 verification`
