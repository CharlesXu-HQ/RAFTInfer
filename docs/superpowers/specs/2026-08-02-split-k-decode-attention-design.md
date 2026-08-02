# Split-K Decode Attention Design

Date: 2026-08-02

Status: Design approved; written specification pending final review

## 1. Decision

RAFTInfer will first optimize long-context Qwen3.5-9B BF16 decode with a
two-stage split-K online-attention path. The existing single-block online
decode kernel remains the short-context and unsupported-shape fallback.

This phase is limited to RTX 50-series GPUs. It does not expand RTX 40-series,
quantized-model, speculative-decoding, or general-model support. After this
phase is implemented and validated, phase B may optimize GEMV/projection
fusion and the complete Gated DeltaNet path under a separate design and plan.

## 2. Motivation

Current RTX 5090 evidence shows that RAFTInfer's generation advantage over the
pinned llama.cpp reference narrows as context length grows:

| Workload | RAFTInfer advantage per generated token |
|---|---:|
| TG128 after PP128 | 359.55 microseconds |
| TG128 after PP512 | 46.10 microseconds |
| TG128 after PP512, confirmation sample | 13.13 microseconds |

Qwen3.5-9B contains 24 Gated DeltaNet blocks and 8 full-attention blocks. Only
full attention scans an increasingly long KV history during decode. The
current decode kernel launches one CUDA block per query head and partitions a
head's history only among warps within that block. For the accepted model shape
this provides 16 primary blocks, which is insufficient parallelism for a long
KV range on RTX 5090.

The phase therefore targets the scaling component most directly correlated
with the observed loss of generation margin.

## 3. Alternatives considered

### 3.1 Two-stage split-K online attention — selected

Multiple CUDA blocks independently scan disjoint KV partitions and emit
partial online-softmax state. A second kernel merges those states per query
head. This preserves online softmax, exposes enough parallelism for the target
GPU, works with ordinary CUDA Graph capture, and has a straightforward
correctness fallback.

### 3.2 Cooperative persistent kernel

A cooperative kernel could combine partition scan and global reduction without
a separate launch. It imposes stronger occupancy and cooperative-launch
constraints, complicates graph capture, and makes fallback and testing less
local. It is not selected for phase A.

### 3.3 Fused KV append and split attention

Fusing cache append can remove a launch and avoid rereading the newest KV
element. It expands ordering, graph, and cache-layout changes beyond the
minimum needed to validate split-K. It remains a later optimization after the
split-K path establishes a measured baseline.

## 4. Kernel design

The first-stage grid is logically indexed by query head and KV partition. Each
block:

1. loads one query head;
2. scans only its assigned logical KV interval from the existing BF16
   head-major cache;
3. computes scores and applies the decode causal bound;
4. maintains FP32 online-softmax state consisting of a local maximum, local
   exponential denominator, and local value accumulator;
5. writes one partial record to fixed scratch storage.

The second-stage grid contains one block per query head. It merges active
partial records using the stable online-softmax merge equations:

```text
global_max = max(partial_max[i])
scale[i] = exp(partial_max[i] - global_max)
global_sum = sum(partial_sum[i] * scale[i])
global_value = sum(partial_value[i] * scale[i])
output = global_value / global_sum
```

The initial partition-size candidates are 256 and 512 KV tokens. They are
compile-time-specialized candidates, selected from target measurements rather
than assumed from theoretical occupancy alone. Accumulation remains FP32 and
the output contract remains identical to the current online kernel.

## 5. Dispatch and fallback

Execution-plan construction records both the existing single-block kernel and
the supported split-K candidates for the accepted Qwen3.5-9B shape. Dispatch
uses a context bucket and the fixed model signature:

- short contexts use the current zero-workspace online kernel;
- long contexts use split-K only after target measurement establishes a
  profitable threshold;
- unsupported head dimensions, dtypes, layouts, or model signatures use the
  existing compatible path;
- invalid tensor or cache contracts continue to return structured errors.

The optimized path must be observable in diagnostic and benchmark metadata.
A fallback run cannot be reported as split-K evidence.

## 6. Workspace and CUDA Graph contract

Split-K relaxes the previous zero-workspace decode rule only for long-context
attention. Scratch storage is:

- bounded by query heads, value-head dimension, and the configured maximum
  partition count;
- allocated once with the execution/session plan through the existing RMM
  ownership boundary;
- held at a fixed address for session lifetime;
- never allocated, resized, or freed inside a decode step.

CUDA Graph topology and addresses remain stable. The captured first-stage grid
uses the maximum partition count for its graph/context bucket. Blocks read the
device-resident logical position, and partitions outside the active KV range
exit without writing an active record. The reduction reads only the active
partition count derived from the same device-resident state.

Phase A does not move per-operator calls across the Rust/C++ boundary. Rust
continues to submit a complete decode step or graph replay through the existing
coarse ABI.

## 7. Correctness requirements

Implementation is not eligible for default dispatch until all of the following
pass:

- partial-state and final-output operator tests against the existing online or
  materialized reference within the repository's accepted BF16/FP32 tolerance;
- test vectors on both sides of every dispatch and partition boundary;
- non-multiple partition tails, minimal valid context, and configured maximum
  context;
- grouped-query head mapping and BF16 head-major KV cache cases used by
  Qwen3.5-9B;
- ordinary-stream and CUDA Graph replay equivalence;
- exact greedy-token equality for all 128 accepted generated tokens.

The existing single-block path remains available as a correctness oracle and
runtime fallback throughout implementation.

## 8. Performance acceptance

Default promotion requires two independent, uncontended RTX 5090 validation
runs using pinned model, prompt, binaries, parameters, warmups, and measured
rounds. Each run must record GPU/process state and selected kernel metadata.

Compared with the current accepted RAFTInfer baseline:

- PP512 generation must improve by at least 1%;
- TG128 after PP512 must improve by at least 1%;
- PP128, PP512 prefill, and short-context generation may not regress by more
  than 1%;
- all correctness gates must pass before performance evidence is accepted.

The pinned llama.cpp comparison is then rerun under the same conditions and
reported, but phase-A promotion is based on an improvement over the accepted
RAFTInfer baseline so that external-reference jitter cannot hide a kernel
regression.

If the numerical path is correct but the performance gate fails, split-K stays
behind an explicit experimental selection and is not the default kernel.

## 9. Implementation boundaries

Expected changes are confined to:

- the Qwen3.5 online-attention CUDA implementation and declarations;
- execution-plan scratch sizing and attention dispatch;
- operator, graph, and end-to-end correctness tests;
- benchmark metadata and verification evidence;
- documentation describing the selected threshold and target results.

Phase A does not change model formats, public Rust request APIs, sampling
semantics, cache dtype/layout, or release correctness policy.
