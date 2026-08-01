# M1 Registry and Correctness Harness Design

Date: 2026-07-24

Status: Approved by the existing project design and the 2026-07-24 execution handoff

## 1. Purpose

M1 builds the contracts that every later Qwen3 Dense kernel must satisfy before
the project loads a full model. It adds tensor metadata, operator signatures,
kernel capabilities, deterministic dispatch, a preallocated workspace arena,
independent CPU references, reusable correctness metrics, and benchmark JSONL.

M1 does not load GGUF, execute a Transformer layer, allocate a KV cache, capture
a CUDA Graph, or import `bw24` source. Those remain later milestones.

## 2. Selected approach

Three implementation orders were considered:

1. Build GPU kernels first and add references afterward.
2. Build all framework abstractions before any real operator.
3. Build the framework as a vertical slice around real CPU reference operators.

M1 uses option 3. Each contract is introduced only when a real operator or test
needs it. This avoids speculative abstractions while establishing independent
oracles before optimized CUDA or imported `bw24` code enters the project.

## 3. Scope and success criteria

M1 is complete when:

- tensor descriptors validate rank, shape, strides, byte bounds, dtype, memory
  type, quantization, and optional scale metadata without dereferencing data;
- operator signatures describe operator kind, prefill/decode regime, tensor
  metadata, shape bucket, graph requirement, determinism requirement, and
  workspace availability;
- kernel capabilities declare supported signatures, alignment and workspace
  requirements, graph safety, determinism, priority, and provenance;
- dispatch is deterministic, caches complete signatures, and exposes explicit
  rejection reasons when no candidate matches;
- a graph-required request never selects a graph-unsafe kernel;
- a deterministic request never selects a nondeterministic kernel;
- the workspace arena performs aligned suballocation from one owned RMM
  allocation and performs no allocation during operator execution;
- independent CPU FP32 references cover RMSNorm, RoPE, BF16 linear, softmax
  plus argmax, embedding, Add, and SwiGLU;
- tests cover deterministic fixtures, seeded randomized inputs, adversarial
  values, invalid metadata, dispatch fallback, and graph-safety behavior;
- correctness reports include maximum absolute error, maximum relative error,
  cosine similarity, and exact-index equality where relevant;
- benchmark records are emitted as versioned JSONL containing commit, device,
  operator signature, selected kernel, correctness state, latency statistics,
  and launch metadata;
- host-only C++ and Rust checks still pass, and the CUDA build passes on the
  shared RTX 5090 target when the existing preflight permits it.

## 4. Component boundaries

### 4.1 Tensor contract

`cpp/include/raftinfer/tensor.h` contains ABI-safe enums and `RaftInferTensorDesc`. The
descriptor uses fixed-size shape and stride arrays with an explicit rank and
byte size. It contains non-owning pointers only. Quantized packing remains
opaque to the execution layer.

Private C++ validation in `cpp/operators/tensor_validation.*` converts failures
into precise messages. Validation never assumes that a non-null device pointer
is host-readable.

### 4.2 Operator and dispatch contract

`cpp/registry/operator_registry.*` owns:

- `OperatorKind`;
- `ExecutionRegime`;
- `OperatorSignature`;
- `KernelCapability`;
- `KernelRegistration`;
- `DispatchResult`;
- `OperatorRegistry`.

The registry contains metadata and callable launch functions but owns no model
weights or GPU allocations. Selection filters incompatible candidates, then
orders matches by priority and stable registration name. The complete signature
is the cache key. A cache hit returns the same immutable registration.

Kernel origin is metadata, not an automatic ranking rule. A validated imported
`bw24` kernel and a project-native kernel compete using the same capability and
performance evidence.

### 4.3 Execution context and workspace

`cpp/execution/execution_context.hpp` is private C++ and exposes the selected
stream, RAFT resources, RMM resource, device properties, and workspace view.
It exists only in CUDA builds.

`cpp/execution/workspace_arena.*` owns one RMM allocation created during engine
or plan setup. `allocate(bytes, alignment)` advances an offset with checked
alignment; `reset()` reuses the allocation. Exhaustion is an explicit error.
Kernel launch functions receive an arena view and may not allocate.

The arena's offset/alignment arithmetic is factored into a host-testable
`WorkspaceLayout` so bounds behavior is verified without CUDA.

### 4.4 CPU reference operators

`cpp/reference/` provides FP32 implementations whose loops and data access are
intentionally simple and independent from future CUDA kernels:

- RMSNorm uses `x / sqrt(mean(x²) + epsilon) * weight`;
- RoPE uses explicit position, base frequency, rotary dimension, and
  interleaved pair ordering;
- BF16 linear converts stored BF16 operands independently and accumulates FP32;
- softmax uses a max-subtracted FP64 host accumulator before FP32 output;
- argmax returns the first index on ties;
- embedding validates every token ID and copies the selected row;
- Add performs elementwise FP32 addition;
- SwiGLU computes `silu(gate) * up`.

Reference APIs use spans and explicit shape structs. They do not call RAFT,
cuBLASLt, CUDA implementations, or code imported from `bw24`.

### 4.5 Correctness reports

`cpp/reference/correctness.*` compares candidate FP32 output with reference
FP32 output. Relative error uses
`abs(candidate-reference) / max(abs(reference), relative_floor)`. Cosine
similarity handles two all-zero vectors as equal and one zero vector as
different. Non-finite mismatches fail explicitly.

Operator tests set tolerances locally. M1 does not define one global epsilon.

### 4.6 Benchmark JSONL

`cpp/benchmarks/benchmark_record.*` serializes one schema-versioned record per
line using a small internal JSON string escaper; M1 adds no JSON dependency.
The schema requires:

- schema version and UTC timestamp;
- project commit;
- device, driver, CUDA, and architecture;
- operator signature and selected kernel;
- provenance kind and upstream revision when applicable;
- correctness pass/fail and metric values;
- warmup count, measured iterations, median, p95, minimum, maximum;
- workspace bytes, launch count, and graph mode.

A record cannot claim publishable performance unless correctness passed.

## 5. Data flow

```text
RaftInferTensorDesc
  -> metadata validation
  -> OperatorSignature
  -> OperatorRegistry::resolve
       -> capability filtering
       -> deterministic ranking
       -> cached DispatchResult
  -> launch with ExecutionContext + preallocated WorkspaceArena
  -> candidate output
  -> independent CPU reference comparison
  -> correctness report
  -> benchmark JSONL (only performance-valid when correctness passed)
```

## 6. Error handling

- Public tensor validation failures use `RAFTINFER_STATUS_INVALID_ARGUMENT`.
- Unsupported signatures use a new `RAFTINFER_STATUS_UNSUPPORTED` status.
- Workspace exhaustion uses `RAFTINFER_STATUS_RESOURCE_EXHAUSTED`.
- A selected kernel launch failure remains a CUDA or internal error according
  to its source.
- Registry resolution returns a structured error containing the complete
  signature and per-candidate rejection reasons.
- No exception crosses the C ABI.
- CPU references reject invalid shapes, indices, epsilon values, or rotary
  dimensions before writing output.

## 7. Testing strategy

Host CTest covers tensor validation, workspace layout, dispatch, references,
correctness metrics, and JSONL validity. Tests use fixed fixtures plus seeded
random generation so failures are reproducible.

Source-contract tests remain limited to ABI properties that cannot be expressed
through normal runtime tests. New behavior is tested through compiled C++ APIs.

The CUDA target check builds the real execution context and workspace arena,
runs the existing smoke, and adds a small allocation/reset probe. It uses the
existing fail-closed shared-GPU preflight and never interferes with unrelated
GPU processes.

Rust receives the new status codes now, but M1 adds no per-operator Rust FFI.
The coarse FFI boundary remains reserved for model/session operations.

## 8. Compatibility and provenance

M1 targets RTX 50-series `sm_120a` only for CUDA builds and remains host-testable
without CUDA. It adds no RTX 40 path.

No `bw24` source is imported in M1. The registry includes provenance fields now
so an already optimized `bw24` kernel can be reused directly in M3 after its
license, pinned revision, modifications, correctness, algorithmic invariants,
and RTX 50 performance evidence pass the common gates.
