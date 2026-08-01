# Architecture

RAFTInfer keeps the host/runtime boundary deliberately coarse:

```text
Rust CLI/runtime
    ↓ coarse C ABI
C++ execution plan and session
    ↓
RAFT resources + RMM ownership
    ↓
cuBLASLt + custom CUDA kernels
```

Rust owns the safe runtime, tokenizer, CLI, and orchestration. C++ owns GPU
state and exposes model/session operations such as creation, prefill, decode,
reset, and logits transfer through the C ABI. Rust never invokes individual
kernels: that would expose device pointers, streams, events, allocation
lifetime, and scheduling details across the language boundary.

The C++ execution plan owns RAFT device resources and RMM-backed allocations.
Sessions use fixed, preallocated workspaces so CUDA Graph capture and replay do
not depend on dynamic allocation in the execution loop. cuBLASLt plans and
project-owned CUDA kernels remain behind this session boundary.

An optimized external kernel, including one with `bw24` provenance, is eligible
for reuse only after its license, source provenance, functional and numerical
correctness, and RTX 50 performance all pass the project gate. Its imported
files and local modifications must be recorded. RAFTInfer does not publish
performance comparisons against `bw24`.
