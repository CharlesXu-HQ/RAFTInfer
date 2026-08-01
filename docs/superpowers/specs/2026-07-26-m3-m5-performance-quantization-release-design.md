# M3-M5 Performance, Quantization, and Release Design

Date: 2026-07-26

Status: Approved amendment to the 2026-07-22 engine design

## 1. Decision

M3-M5 completes the first releasable Qwen3.5-9B text engine while preserving the
original architecture:

- RAFT and RMM remain the CUDA foundation and resource-management layer.
- Rust remains the public runtime and CLI layer.
- C++ owns model execution plans and coarse-grained CUDA submission.
- Performance-critical CUDA operators are project-owned or reused from `bw24`
  when their format, mathematical semantics, license, correctness, and measured
  performance are compatible.
- GGML and the llama.cpp runtime are not adopted. llama.cpp is a pinned
  correctness and performance reference and a source of applicable kernel and
  scheduling techniques.

The implementation order is deliberately performance-first:

1. remove the known BF16 execution bottlenecks;
2. add Q4_K_M without hiding those bottlenecks;
3. tune the shared runtime for both paths;
4. close release, reproducibility, and provenance gates.

The first release targets RTX 50-series GPUs only. RTX 40-series support is
outside the v0.1 acceptance scope.

## 2. Current evidence and problem statement

The completed M2 baseline is functionally correct and has exact greedy-token
parity on the fixed acceptance corpus, but its performance is uneven on the RTX
5090 target:

| Workload | RAFTINFER tok/s | llama.cpp tok/s | Ratio |
|---|---:|---:|---:|
| PP128 | 6065.34 | 3435.46 | 1.7655 |
| TG128 after PP128 | 82.00 | 84.23 | 0.9735 |
| PP512 | 5469.65 | 6612.42 | 0.8272 |
| TG128 after PP512 | 70.60 | 84.12 | 0.8393 |

The measurements identify two different problems:

- longer prefill loses bandwidth and launch efficiency, particularly in full
  attention;
- decode loses to accumulated launch, dispatch, cache-traffic, and small-matrix
  overhead.

The current full-attention implementation materializes a complete logits
workspace and launches separate logits, softmax, and value-aggregation kernels.
The current executor does not capture decode in a CUDA Graph. cuBLASLt planning
uses a general heuristic rather than a token-shape-specific decode/prefill
policy. The full-attention KV cache is FP32.

The existing Gated DeltaNet implementation already keeps recurrent state shards
resident in registers during its main loop. It remains the starting point and
is not replaced merely to match another project's source organization.

## 3. Acceptance contract

BF16 and Q4_K_M are independent release configurations. Each configuration
must satisfy all of the following against the same pinned llama.cpp revision,
same GGUF contents, same prompt tokens, same generated-token count, and same
RTX 50 target:

- PP128 ratio is at least `1.0x`;
- PP512 ratio is at least `1.0x`;
- TG128 after PP512 ratio is at least `1.0x`;
- at least one of those three ratios is at least `1.1x`;
- every required operator passes its numerical correctness threshold;
- every fixed-corpus generation run has exact greedy-token equality.

The `1.1x` requirement applies independently to BF16 and Q4_K_M. One strong
BF16 result cannot satisfy the Q4_K_M gate, or vice versa.

TG128 means generating 128 greedy tokens after the accepted 512-token prefill.
TG128 after PP128 is also recorded as a diagnostic measurement but does not
replace the stricter PP512-context release gate.

NVFP4 is an optional v0.1 enhancement. It may be added when a compatible bw24
path can be imported cleanly and validated, but it cannot replace either
mandatory BF16 or Q4_K_M acceptance.

Performance evidence is invalid when:

- the tested path falls back to an unreported reference implementation;
- the target GPU is materially occupied by an unrelated process;
- correctness gates for that configuration have not passed;
- model, prompt, generation, cache, or sampling settings differ;
- the benchmark does not complete its configured warmup and measured rounds.

## 4. Runtime and ABI boundary

Rust continues to submit work at request-step granularity:

```text
load_model(...)
create_session(...)
prefill(session, token_ids)
decode_step(session, params)
reset_session(session)
```

Rust does not call individual attention, projection, normalization, or state
update operators. A single Rust-to-C++ call submits a complete prefill request,
a complete decode step, or a CUDA Graph replay.

The C++ model owns immutable plan data:

```text
ModelExecutionPlan {
  block plans,
  tensor bindings,
  kernel selections,
  matrix algorithm selections,
  workspace layouts,
  graph-compatible parameter storage,
  prefill buckets,
  decode graph variants
}
```

The session owns mutable state:

```text
SessionExecutionState {
  logical position,
  device-resident length and offset scalars,
  convolution state,
  recurrent state,
  full-attention KV cache,
  fixed-address token and logits buffers,
  captured graph instances
}
```

RMM allocations are completed before graph capture. Decode performs no dynamic
allocation. Tensor descriptors, cuBLASLt descriptors, algorithm selection, and
workspace addresses are reused rather than rebuilt per token.

## 5. Kernel selection contract

The registry chooses an implementation using semantic and performance-relevant
capabilities:

```text
KernelKey {
  operator,
  compute capability,
  activation dtype,
  weight format,
  accumulation dtype,
  M/N/K or attention shape,
  token bucket,
  head dimension,
  cache dtype and layout,
  causal/decode mode
}
```

Quantization unpacking is an implementation detail of a selected quantized
kernel, but quantization semantics are part of the key. A Q4_0, NVFP4, or
Q4_K_M kernel cannot be selected for another format merely because tensor
dimensions match.

Every benchmark record includes the selected kernel family, graph mode, cache
dtype, matrix path, model fingerprint, and reference revision. Correctness
fallbacks remain available for unsupported shapes, but release performance
gates require the intended optimized path.

## 6. Online-softmax full attention

The full logits workspace is removed from the optimized attention path.

The new prefill kernel:

- tiles query and key/value regions;
- loads Q/K/V through coalesced vector operations;
- maintains row maximum, exponential denominator, and output accumulator
  online;
- applies causal masking before each online update;
- accumulates numerically sensitive state in FP32;
- writes only the final normalized attention output;
- specializes common Qwen3.5 head and tile dimensions;
- selects tile shapes using measured RTX 50 occupancy and register pressure.

The decode kernel:

- treats the single query separately from the prefill path;
- streams the logical KV range without materializing logits;
- uses split-K only when context length makes the extra reduction profitable;
- keeps graph-compatible argument addresses stable;
- fuses KV append when doing so preserves ordering and improves measured
  performance.

The implementation may reuse scheduling patterns from llama.cpp's tiled
FlashAttention implementation, subject to license and provenance requirements,
but it is adapted to RAFTINFER's tensor layout and execution plan rather than wrapped
through GGML.

An unoptimized reference attention path remains test-only and diagnostic. It is
not eligible for the performance gate.

## 7. Decode CUDA Graph

Decode graph variants are captured after the session has fixed buffer
addresses. A graph variant is keyed by the state that changes graph topology,
not by the current token value:

```text
DecodeGraphKey {
  batch bucket,
  context bucket,
  cache dtype/layout,
  weight format,
  enabled operator families
}
```

Token id, logical position, KV length, recurrent offsets, and result scalars
live at fixed device addresses. A normal decode step updates those small inputs
and launches one graph replay.

The graph manager:

- captures only after all required lazy plans have been materialized;
- reports capture and replay status in benchmark metadata;
- invalidates a graph when topology-affecting session configuration changes;
- supports ordinary stream execution as a correctness fallback;
- never silently labels ordinary execution as graph replay.

TG128 release measurements require graph replay unless a measured, documented
alternative is faster for the exact release shape.

## 8. Gated DeltaNet

The current register-resident recurrent-state kernel is retained and tuned.
The next optimization pass evaluates:

- four-warps-per-block column ownership;
- transposed persistent-state layout for coalesced boundary loads/stores;
- `__launch_bounds__` and register-count control;
- compile-time specializations for key/value dimensions including 64 and 128;
- distinct prefill and single-token decode schedules;
- fused state-cache update or snapshot behavior where it reduces traffic.

An applicable bw24 or llama.cpp-derived implementation is reused only if:

1. it implements the same Gated DeltaNet equations and state transition;
2. its state layout can be adopted without breaking the session contract;
3. its source license is compatible and preserved;
4. layered correctness passes;
5. it outperforms the existing kernel on the target workload.

Passing provenance or performance alone is insufficient.

## 9. KV cache dtype and layout

Two full-attention cache modes are retained:

- FP32 reference cache for diagnosis and numerical comparison;
- BF16 optimized cache for release candidacy.

The optimized cache design uses a layout chosen for coalesced append and
attention reads. Token-major and head-major candidates are benchmarked rather
than selected by convention. Vectorized BF16 loads widen to FP32 for online
softmax accumulation.

KV append, cache-length advancement, attention consumption, and failure
reporting preserve atomic session semantics. A failed step cannot expose a
partially advanced logical position.

BF16 cache becomes the release default only after:

- cache append/read unit tests;
- attention output comparison against the FP32-cache reference;
- continued-prefill and decode state-transition tests;
- fixed-corpus exact greedy-token parity;
- a measured performance improvement or memory-traffic justification.

## 10. Shape-specific matrix execution

Matrix execution is divided by token shape and weight format.

BF16:

- prefill uses cached cuBLASLt algorithms selected per representative token
  bucket;
- decode uses GEMV or small-M GEMM paths selected independently;
- algorithm descriptors and workspaces are created once;
- common projection shapes may receive project CUDA kernels when they beat the
  cached library path.

Q4_K_M:

- the loader validates exact GGUF block sizes and tensor byte spans;
- the upload plan preserves or explicitly repacks scale, minimum, and quantized
  value fields;
- decode uses fused dequantization and matrix-vector execution;
- prefill uses a measured quantized GEMM path, with activation quantization only
  when its numerical gate passes;
- split-K is enabled only for shapes where its reduction overhead is recovered.

Small compatible projections may be grouped only when the grouping does not
change model equations, tensor ordering, or graph stability.

## 11. bw24 reuse policy

The official bw24 repository is treated as a performance-kernel upstream, not
as a second runtime dependency.

For each candidate:

1. record upstream repository, commit, source path, license, and local adapter;
2. identify the exact operation, format, layout, supported shapes, and numeric
   assumptions;
3. test the unmodified or minimally adapted kernel against the independent
   project reference;
4. benchmark it against the existing RAFTINFER implementation on RTX 50;
5. reuse it directly when it is compatible and faster;
6. retain attribution and document material modifications.

Optimized bw24 code is not rewritten solely for project ownership. Conversely,
an NVFP4 or Q4_0 kernel is not claimed as Q4_K_M support. Shared scheduling,
packing, or graph techniques may be adapted while keeping the formats distinct.

## 12. Correctness strategy

Correctness remains layered:

1. format and tensor-layout validation;
2. CPU FP32 operator references;
3. BF16/quantized operator error thresholds;
4. per-block state and hidden-output checks;
5. continued-prefill, decode, and reset transition checks;
6. tokenizer parity;
7. exact greedy-token equality at every generated step.

New performance work receives targeted tests before its optimized
implementation:

- online-softmax versus materialized attention;
- causal boundary and long-context numerical stability;
- graph replay versus ordinary decode state and logits;
- graph invalidation and reset;
- FP32 versus BF16 KV cache;
- Q4_K_M unpacking, dequantization, matvec, and GEMM;
- selected-kernel metadata and forbidden fallback detection;
- imported bw24 kernel compatibility fixtures.

Performance improvements are never accepted by weakening exact greedy-token
requirements. If a BF16 cache or quantized activation path changes a greedy
token, it remains experimental until the discrepancy is understood and fixed.

## 13. Benchmark methodology

The acceptance runner performs:

1. target identity and CUDA/device capability checks;
2. GPU-process and utilization preflight;
3. model and binary fingerprint capture;
4. correctness gate execution;
5. at least three warmup rounds;
6. seven measured rounds;
7. median and dispersion calculation;
8. RAFTINFER/llama ratio evaluation;
9. structured artifact emission.

The runner uses a pinned llama.cpp commit and records its build flags. RAFTINFER and
llama.cpp run sequentially on the same target with matching weight format, KV
cache dtype, prompt tokens, context, and generation settings. Throughput gates
compare the median of seven measured rounds. A coefficient of variation above
three percent triggers a new GPU preflight and invalidates the affected result
if the instability persists. A run affected by unrelated GPU activity is
marked invalid rather than averaged into the result.

The fixed mandatory workloads are:

- PP128;
- PP512;
- TG128 after PP512.

TG128 after PP128 and additional context lengths are diagnostic and do not
replace the fixed gates. Peak RMM allocation, CUDA Graph status, cache dtype,
kernel selection, and per-stage timings are captured with throughput.

## 14. Delivery slices

### M3: Q4_K_M vertical path

- exact GGUF Q4_K_M validation and upload/repack;
- Q4_K_M projection kernels and dispatch;
- CPU/operator/end-to-end correctness;
- same-quant llama.cpp benchmark support;
- direct reuse of compatible, faster bw24 kernels with provenance.

### M4: shared execution optimization

- tiled online-softmax prefill and decode attention;
- removal of optimized-path logits workspace;
- decode CUDA Graph;
- BF16 KV-cache candidate;
- token-shape-specific matrix dispatch;
- additional Gated DeltaNet scheduling work;
- structured stage and kernel-selection telemetry.

Implementation may deliver the BF16 subset of M4 before completing M3 so that
Q4_K_M does not conceal shared runtime bottlenecks. Milestone acceptance still
requires each slice's complete gates.

### M5: release hardening

- deterministic build and benchmark instructions;
- automated BF16 and Q4_K_M release gates;
- dependency and imported-source provenance;
- target capability and fallback diagnostics;
- public limitations and supported configuration;
- consolidated correctness, performance, and memory evidence;
- v0.1 release checklist.

## 15. Failure handling and observability

Public failures use structured status codes and include the failing model,
operator, format, shape, and selected execution path when safe to report.

The runtime exposes diagnostic metadata for:

- selected attention implementation;
- selected matrix implementation;
- CUDA Graph capture/replay/fallback;
- KV cache dtype and layout;
- quantization format and repack version;
- fallback reason;
- peak RMM allocation.

Diagnostics are collected outside timing regions or through low-overhead device
events. Debug instrumentation cannot remain enabled in release performance
measurements.

## 16. Deferred scope

The following remain outside mandatory v0.1 acceptance:

- RTX 40-series support;
- multimodal vision execution;
- MTP/speculative decoding;
- multi-request continuous batching;
- paged KV-cache allocation;
- NVFP4 as a mandatory format;
- distributed or multi-GPU execution;
- compatibility with arbitrary GGML execution graphs.

These exclusions do not prevent small forward-compatible interfaces, but they
cannot delay the BF16 and Q4_K_M release gates.

## 17. Completion condition

M3-M5 is complete only when:

- both mandatory configurations pass layered correctness;
- both configurations meet their independent three-workload performance gates;
- no required measurement relies on an undisclosed fallback;
- target evidence is reproducible from documented commands;
- reused bw24 source has complete provenance and attribution;
- all local tests, target CUDA tests, Rust checks, and release scripts pass;
- the README and release evidence accurately describe supported and deferred
  behavior.
