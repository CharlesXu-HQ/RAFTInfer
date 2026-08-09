# Decode GEMV Phase B1 Design

Date: 2026-08-09

Status: Approved in conversation; written specification pending user review

## 1. Decision

RAFTInfer will begin phase B with an independently measurable decode-matrix
subphase, B1. B1 optimizes only Qwen3.5-9B BF16, batch-one, single-token
decode on RTX 50-series GPUs. Its formal release target is the RTX 5090.

For a matrix operation

```text
C[M, N] = A[M, K] * W[K, N]
```

B1 implements and evaluates project-owned or compatibly reused kernels only
for `M = 1`. This is the matrix-vector, or GEMV, shape used by the current
batch-one greedy decode path. The interfaces may represent future `M = 2` or
`M = 4` plans, but B1 does not implement, benchmark, or promote those shapes.

Existing tuned cuBLASLt plans remain the correctness, performance, and
fallback baseline. Prefill buckets such as `M = 128` and `M = 512` remain on
cuBLASLt and are outside B1.

## 2. Scope and non-goals

### 2.1 In scope

- BF16 weights and BF16 matrix inputs for Qwen3.5-9B decode;
- FP32 accumulation with output dtype matching the existing plan contract;
- the LM head, FFN gate/up/down, full-attention projections, and linear-
  attention projections reached by `M = 1` decode;
- per-shape comparison among tuned cuBLASLt, compatible bw24 kernels, and
  RAFTInfer CUDA GEMV kernels informed by an audited llama.cpp revision;
- immutable plan selection before CUDA Graph capture;
- fixed workspace allocation through the existing RMM-backed executor arena;
- C++, C ABI, Rust, CLI, parity, benchmark, and release-gate diagnostics;
- exact greedy-token, layered numerical, CUDA Graph, allocation, and RTX 5090
  performance verification.

### 2.2 Non-goals

- RTX 40-series support;
- `M > 1` small-M kernels, batched decode, or speculative decode;
- quantized GEMV, including Q4_K_M and NVFP4;
- prefill GEMM replacement;
- RMSNorm, activation, projection, or residual fusion;
- Gated DeltaNet state-transition fusion;
- KV append or attention fusion;
- silently changing stored weight layouts.

Grouped/fused projections, safe RMSNorm fusion, and complete Gated DeltaNet
launch and traffic fusion remain phase B2, B3, and B4 respectively. Keeping
them outside B1 makes numerical and performance changes attributable to the
matrix path alone.

## 3. Existing execution contract

The Qwen3.5 executor currently creates and caches cuBLASLt plans for token
buckets 1, 128, and 512. Plans with identical matrix configuration are shared
across layers. Construction performs bounded cuBLASLt candidate tuning, while
execution reuses the selected algorithm, fixed input-conversion buffer, fixed
output buffers, and fixed workspace.

Decode consumes FP32 layer activations. `run_matmul_group` converts a shared
input once to the weight dtype and then issues the projection plans. CUDA Graph
captures the resulting fixed-address decode sequence. B1 must preserve these
properties:

- one shared conversion for projections with the same input;
- no allocation, descriptor creation, environment lookup, or plan selection
  during decode or graph replay;
- row-major input, stored weight orientation, output dtype, and buffer-size
  contracts;
- atomic session position and model-state transitions.

## 4. Architecture

### 4.1 Decode matrix plan

B1 introduces a decode-only matrix-plan abstraction whose public behavior is
equivalent to the current cuBLASLt plan for `M = 1`. A plan owns immutable
shape, dtype, implementation, launch, workspace, and provenance metadata. It
does not own model weights, input buffers, output buffers, or per-step state.

The implementation variants are:

1. `cublaslt`: the current tuned plan and permanent fallback;
2. `bw24`: a source-audited, license-compatible kernel whose equations,
   precision, layout, alignment, and shape contract match RAFTInfer exactly;
3. `raftinfer`: a project CUDA kernel whose scheduling may use techniques from
   an audited, pinned llama.cpp revision.

The executor plan cache remains keyed by the complete matrix signature. Layers
with identical signatures share the selected decode plan. Prefill plans remain
ordinary `CublasLtMatmulPlan` instances.

### 4.2 Candidate admission

A bw24 candidate is admitted only when its source is available for audit and
its provenance and license are recorded. Direct reuse requires exact semantic
and layout compatibility; an adapter may translate the RAFTInfer launch
interface but may not insert a hidden repack, per-step allocation, or numerical
transformation. If those conditions do not hold, the design may reuse the
scheduling idea but must identify the resulting kernel as RAFTInfer code.

llama.cpp is a pinned scheduling reference rather than a runtime dependency.
Vectorized loads, warp assignment, reduction structure, and output tiling may
be adapted after their source revision and license are recorded. Directly
transferred code retains required attribution.

No candidate is described as reused until its source and compatibility have
been verified.

### 4.3 Shape families

Different width ratios are allowed to select different kernel families:

- large-output shapes, including the LM head and FFN gate/up;
- large-input shapes, including FFN down;
- medium attention and linear-attention QKV, gate, and output projections;
- narrow beta, alpha, key, and value projections.

A project kernel may specialize tile dimensions, warps per block, vector
width, reduction strategy, and launch bounds per accepted shape family. B1
does not require a single general kernel to cover every shape.

## 5. Construction-time selection

The execution policy supports `auto`, `cublaslt`, and `custom` decode-matrix
modes.

- `auto` constructs all eligible candidates and selects per shape. It is the
  formal default only after the promotion gate passes.
- `cublaslt` forces the baseline for parity and performance comparisons.
- `custom` requires an admitted non-cuBLASLt candidate for every signature in
  an explicit immutable candidate allowlist. Signatures outside that allowlist
  remain visibly declared cuBLASLt plans. A missing custom implementation for
  an allowlisted signature is a structured construction error, never a silent
  fallback.

Selection occurs before decode graph capture:

1. create the tuned cuBLASLt baseline;
2. construct compatible custom candidates without changing model weights;
3. fill fixed scratch input with deterministic nonzero data;
4. compare each candidate output against cuBLASLt under the operator numerical
   threshold; independent CPU FP32 comparison is performed by the required
   offline operator tests rather than by every model construction;
5. warm each passing candidate and collect multiple CUDA-event samples;
6. reject non-finite timing data or a coefficient of variation above 3%;
7. select a custom candidate only when its median is at least 3% faster than
   cuBLASLt;
8. freeze implementation, launch parameters, addresses, and workspace before
   graph capture.

The selected custom kernel may use zero workspace. Any nonzero custom
workspace is included in the one-time executor workspace estimate and receives
a stable aligned address from the existing arena. Selection itself must not
change the post-construction RMM allocation peak.

## 6. Decode data flow

The B1 decode data flow is:

```text
FP32 layer activation
  -> existing shared FP32-to-BF16 cast
  -> immutable selected decode matrix plan
  -> existing BF16 output buffer with FP32 accumulation semantics
  -> existing attention, DeltaNet, SwiGLU, residual, or sampling operation
```

B1 preserves grouped input conversion. For example, FFN gate and up consume
one converted input and then invoke their independently selected plans. B1
does not fuse those launches or their downstream activation.

CUDA Graph captures calls to the already selected implementation. Replay does
not execute a host implementation branch, retune candidates, create CUDA
objects, parse environment variables, or change workspace addresses.

## 7. Fallback and failure semantics

In `auto`, the executor retains cuBLASLt for a shape when:

- the shape, dtype, layout, alignment, or target capability is unsupported;
- source, license, or provenance requirements are not satisfied;
- numerical comparison fails;
- the median advantage is below 3%;
- timing CV exceeds 3% or timing is non-finite;
- fixed workspace or CUDA Graph requirements cannot be satisfied.

All construction-time rejections are recorded. They do not make a valid
session unavailable when the cuBLASLt baseline remains usable.

After a plan is frozen, a runtime kernel failure does not retry the operation
with cuBLASLt. The session is poisoned, the call returns a structured CUDA or
execution error, and its logical token and position are not committed.
Physical KV, convolution, or recurrent buffers may have been partially written
and are considered invalid until a complete reset or session recreation. This
prevents a partially executed step from being applied twice or exposed as a
valid committed state.

## 8. Diagnostics and ABI

Internal diagnostics retain one record per unique decode matrix signature:

- requested policy and selected implementation;
- `M`, `N`, `K`, dtypes, layout, and required alignment;
- implementation source and kernel identifier;
- source or kernel digest and reference revision;
- baseline and selected medians, CV, relative speedup, and workspace bytes;
- numerical result and fallback or rejection reason;
- whether CUDA Graph captured the selected custom kernel.

The stable C ABI diagnostics struct is extended only at its tail and mirrored
in `raftinfer-sys`, the safe Rust runtime, and CLI JSON. The native function is
changed from exact-size-only validation to recognizing the existing prefix and
the new full layout; it writes only fields present in the caller-declared
`struct_size`. The public summary discloses requested mode, resolved overall
implementation (`cublaslt`, `custom`, or `mixed`), custom and cuBLASLt shape
counts, kernel/source digest, fixed workspace bytes, graph capture state, and
whether any fallback occurred. Detailed benchmark evidence retains the
per-shape records.

Tests lock enum values, field order, struct size handling, Rust translation,
and exact CLI JSON names. No benchmark result is accepted when required matrix
diagnostics are missing or inconsistent.

## 9. Correctness verification

### 9.1 Operator tests

Every candidate shape is tested with deterministic nonzero, seeded random,
extreme finite, and alignment-boundary inputs. Results are compared with both
the independent CPU FP32 operator and cuBLASLt BF16 baseline. FP32
accumulation must satisfy the layered operator threshold. NaN, infinity,
out-of-bounds writes, undersized workspace, wrong dtype, wrong device, and
unsupported alignment fail explicitly.

### 9.2 Executor tests

Forced `custom`, forced `cublaslt`, and `auto` executions begin from identical
model, prompt, and state. Tests compare layer localization traces, final
logits, greedy token IDs, position, full-attention KV cache, linear convolution
state, recurrent state, reset, and reuse. Exact floating-point identity with
cuBLASLt is not required; fixed-corpus greedy token IDs must be exact.

Repeated prefill and decode after construction must leave the RMM logical peak
unchanged.

### 9.3 CUDA Graph tests

Each promoted custom plan must capture, instantiate, and replay. Graph and
ordinary-stream paths compare final logits, token IDs, position, KV cache,
convolution state, and recurrent state. Test instrumentation proves that graph
replay performs no plan selection, descriptor construction, allocation, or
environment lookup and retains all recorded addresses.

## 10. Performance and promotion gate

Formal measurements use an idle RTX 5090, the pinned Qwen3.5-9B BF16 model,
the pinned llama.cpp reference, immutable artifact identities, and the existing
fail-closed preflight and provenance rules.

Each custom shape must first beat its tuned cuBLASLt microbenchmark by at least
3% with CV at most 3%. End-to-end promotion then interleaves:

1. a forced-cuBLASLt baseline;
2. independent `auto` candidate A;
3. independent `auto` candidate B.

Both candidate samples must pass layered numerical checks, exact fixed-corpus
greedy-token parity, BF16 disclosure, implementation and digest identity,
CUDA Graph disclosure, coefficient-of-variation limits, and uncontested GPU
preflight/postflight checks.

Relative to the same-window RAFTInfer cuBLASLt baseline:

- PP128, PP512, and TG128 may not regress by more than 1%;
- at least one generation workload must improve by at least 3% in both
  independent candidate samples.

Promotion is per shape. A slow, unstable, or unsupported shape remains on
cuBLASLt even when another shape is promoted. Only accepted target evidence
may change the `auto` allowlist, README performance chart, benchmark artifact,
or release verification document.

If no custom shape produces the required end-to-end generation improvement,
B1 does not change the formal default. Experimental code is retained only when
it remains useful, isolated, tested, and clearly non-default; otherwise it is
removed. The project then records the rejected gate and may proceed to B2.

## 11. Implementation sequence

The implementation plan will preserve causal isolation in this order:

1. add decode-matrix policy and diagnostics contracts with host RED tests;
2. add operator reference, buffer, error, and candidate-selection test seams;
3. inventory and audit exact bw24 and pinned llama.cpp candidate sources;
4. add the first RAFTInfer or compatible bw24 kernel for one measured shape
   family;
5. integrate immutable per-shape plans while preserving grouped casts;
6. add CUDA Graph, no-allocation, state, FFI, Rust, CLI, and script gates;
7. run target microbenchmarks and retain only passing shape candidates;
8. run forced baseline and two independent end-to-end candidate samples;
9. promote only accepted shapes and publish verified evidence.

Each implementation task begins with a failing focused test and ends with the
smallest relevant local or target verification before the next task starts.
