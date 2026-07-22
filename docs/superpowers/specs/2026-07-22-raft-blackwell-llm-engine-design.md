# RAFT-Based Blackwell LLM Engine Design

Date: 2026-07-22

Status: Approved design

## 1. Purpose

Build an independent open-source LLM inference engine for NVIDIA RTX 50-series consumer Blackwell GPUs. The engine uses RAFT and RMM as its common GPU foundation while retaining or developing custom CUDA/CUTLASS kernels for performance-critical paths.

The project is not a generic CUDA framework and is not a simple fork of `bw24`. It reuses proven `bw24` kernels when they match the requirements, but rebuilds the runtime around explicit RAFT/RMM resource ownership, a C++ operator registry, a stable C ABI, and a safe Rust control layer.

The v0.1 reference workload is Qwen3.5-9B Dense in GGUF format, with F16/BF16 as a correctness baseline and Q4_K_M as the first optimized quantized path.

## 2. Goals

- Support RTX 50-series GPUs only in v0.1, targeting consumer Blackwell `sm_120a`.
- Use RAFT/RMM for device resources, streams, events, memory pools, workspaces, tensor views, and selected non-hot-path primitives.
- Use custom CUDA/CUTLASS kernels for quantized linear algebra, attention, fused normalization/activation, and other hot paths.
- Use a C++20/CUDA execution core with a safe Rust runtime and CLI.
- Complete end-to-end Dense-model inference: model loading, prefill, KV cache, decode, sampling, and CLI generation.
- Make kernel functional correctness, algorithmic correctness, and performance equal release gates.
- Match llama.cpp greedy token output on the reference workload.
- Achieve single-request decode performance no lower than llama.cpp and exceed it by at least 10% in one published scenario.
- Preserve complete provenance for imported code and published benchmark results.

RTX 50-series support means one `sm_120a` backend and capability-based launch tuning across the family; it does not mean that every model/dtype combination fits every card. Qwen3.5-9B F16/BF16 validation requires a card with sufficient VRAM, while Q4_K_M is the required lower-memory delivery path. The compatibility matrix must report model, format, maximum context, peak memory, and tested GPU rather than making a blanket model-support claim.

## 3. Non-goals for v0.1

- RTX 40-series or older NVIDIA architectures.
- Multi-GPU execution.
- Continuous batching.
- MoE execution or expert caching.
- Speculative decoding.
- Quantized or paged KV cache.
- Expert or model spilling to host RAM or NVMe.
- Safetensors loading.
- HTTP/OpenAI-compatible service.
- Python bindings.
- A dynamic kernel compiler.

## 4. Architecture

```text
Rust Application Layer
├── CLI
├── Tokenizer and chat templates
├── request/session API
├── configuration and user-facing errors
└── benchmark driver
          │
          │ stable, coarse-grained C ABI
          ▼
C++ Execution Engine
├── Engine
│   ├── device initialization
│   ├── RAFT resources
│   └── RMM memory pool
├── Model
│   ├── GGUF loader
│   ├── tensor catalog
│   ├── model execution plan
│   └── GPU weight storage
├── Session
│   ├── KV cache
│   ├── token/context state
│   └── CUDA Graph instance
├── Executor
│   ├── prefill
│   ├── decode
│   └── graph capture/replay
└── Operator Registry
    ├── imported or project-native custom CUDA/CUTLASS kernels
    ├── cuBLASLt fallback
    ├── RAFT baseline/fallback
    └── reference implementations
          │
          ▼
GPU Foundation
├── RAFT resources, views, and selected primitives
├── RMM device, pinned, and workspace memory
├── CUDA Runtime/Driver
├── CUTLASS
└── cuBLASLt
```

### 4.1 Ownership

- `Engine` exclusively owns one device context, one RAFT resource set, and one RMM memory pool.
- `Model` belongs to an `Engine` and owns uploaded model weights and immutable execution plans.
- `Session` belongs to a `Model` and owns its KV cache, context state, RNG state, workspace bindings, and CUDA Graph instances.
- Rust owns CLI state, tokenization, request/session safety wrappers, configuration, and user-facing errors.
- C++ owns GGUF weight mapping, GPU allocations, streams, events, graphs, kernel dispatch, and GPU error recovery.
- Rust never owns or dereferences a GPU pointer.
- C++ exceptions never cross the ABI. They become structured status codes plus retrievable error details.
- RAFT, RMM, CUTLASS, and C++ template types never appear in the public ABI.

One engine may load multiple models at the API level, but v0.1 activates only one model at a time and executes one session at a time. The public types must not prevent later multi-session scheduling, but v0.1 contains no concurrency scheduler.

### 4.2 Repository structure

```text
/
├── CMakeLists.txt
├── Cargo.toml
├── cpp/
│   ├── include/brt/
│   │   ├── c_api.h
│   │   ├── tensor.h
│   │   └── status.h
│   ├── foundation/
│   ├── kernels/
│   │   └── imported/bw24/
│   ├── operators/
│   ├── registry/
│   ├── model/
│   ├── execution/
│   └── tests/
├── rust/
│   ├── brt-sys/
│   ├── brt-runtime/
│   └── brt-cli/
├── benchmarks/
├── tests/
│   ├── fixtures/
│   └── parity/
├── docs/
│   └── provenance/
└── THIRD_PARTY_LICENSES/
```

`brt` (Blackwell RAFT Runtime) is the stable technical namespace and ABI prefix. A later public repository name does not change this namespace.

## 5. C++/Rust ABI

The ABI uses opaque handles and coarse-grained operations:

```cpp
typedef struct EngineHandle EngineHandle;
typedef struct ModelHandle ModelHandle;
typedef struct SessionHandle SessionHandle;

BrtStatus brt_engine_create(const BrtEngineConfig*, EngineHandle**);
BrtStatus brt_engine_load_model(EngineHandle*, const char* gguf_path, ModelHandle**);
BrtStatus brt_model_copy_tokenizer_spec(ModelHandle*, BrtOwnedBuffer*);
BrtStatus brt_session_create(ModelHandle*, const BrtSessionConfig*, SessionHandle**);
BrtStatus brt_session_prefill(SessionHandle*, const int32_t*, size_t, BrtPrefillResult*);
BrtStatus brt_session_decode(SessionHandle*, const BrtDecodeParams*, BrtDecodeResult*);
BrtStatus brt_session_reset(SessionHandle*);

void brt_owned_buffer_free(BrtOwnedBuffer*);
void brt_session_destroy(SessionHandle*);
void brt_model_destroy(ModelHandle*);
void brt_engine_destroy(EngineHandle*);
```

Rust wraps the C ABI in RAII types. Normal generation transfers token IDs and small result structs, not full logits or GPU buffers. Full logits are available only in explicit diagnostic mode.

The C++ GGUF loader is the single authority for model and tokenizer metadata. After model loading, `brt_model_copy_tokenizer_spec` returns a versioned, owned tokenizer-spec buffer containing the vocabulary, merges, special-token IDs, normalization rules, and chat-template metadata required by Rust. Rust copies/parses the buffer and releases it with `brt_owned_buffer_free`; it does not reopen and independently reinterpret the GGUF file.

One `session_decode` call performs the complete decode step, including graph replay or ordinary execution, logits processing, sampling, KV state update, and device counter update. Rust does not call C++ once per layer or operator.

## 6. Tensor and operator contracts

### 6.1 Tensor descriptions

The execution layer understands the semantic description of a quantized tensor but does not implement its physical unpacking.

```cpp
struct TensorDesc {
    void* data;
    DataType dtype;
    QuantFormat quant;
    MemoryType memory;
    std::array<int64_t, 4> shape;
    std::array<int64_t, 4> strides;
    uint32_t rank;
    const void* scales;
    const void* zero_points;
    size_t byte_size;
};
```

The descriptor exposes enough metadata for validation and dispatch. Format-specific block structures, bit packing, scaling, and dequantization remain in the relevant kernel module and its independent reference implementation.

### 6.2 Operator requests

An `OpRequest` identifies the operator, prefill/decode regime, input/output tensors, and semantic attributes. The registry resolves it by:

- operator type;
- activation, weight, and output data types;
- quantization format;
- tensor layout and alignment;
- problem-shape bucket;
- target architecture;
- prefill or decode regime;
- graph mode and graph-safety requirement;
- workspace availability;
- determinism requirement.

The resolution result is cached by the complete signature.

### 6.3 Kernel capabilities

Every kernel declares:

- supported operator and regime;
- architecture;
- dtype and quantization combinations;
- shape, stride, and alignment constraints;
- required workspace;
- graph safety;
- determinism;
- priority;
- provenance metadata.

Kernels receive an `ExecutionContext` containing RAFT resources, the RMM memory resource, the selected CUDA stream, the preallocated workspace arena, and device properties. A kernel must not create its own stream or allocate memory during execution.

### 6.4 Selection and fallback

The normal preference order is:

1. A proven imported `bw24` kernel or project-native `sm_120a` kernel, ranked by measured fitness rather than origin.
2. A Blackwell-specialized CUTLASS implementation.
3. cuBLASLt.
4. A RAFT-composed baseline for non-hot paths.
5. A simple reference implementation.

Fallback is allowed during planning, initialization, or first-run validation. A captured graph cannot change its kernel bindings during replay. A kernel that fails self-validation is disabled for the process lifetime and the execution plan is rebuilt. Unsupported semantics produce an explicit error; the engine never silently substitutes a different dtype or layout.

## 7. Model execution

### 7.1 Loading and planning

The C++ GGUF loader validates the file version, architecture, metadata, tensor names, tensor counts, shapes, offsets, block alignment, quantization layout, and memory budget. It maps the file, validates tensors, uploads or repacks weights, and constructs immutable prefill and decode plans.

F16/BF16 weights provide the correctness baseline. Q4_K_M is validated and, where required, repacked into the layout expected by the selected kernel. Uploaded weights retain fixed addresses.

Loading fails atomically: missing or invalid core tensors do not produce a partially usable model.

### 7.2 Prefill

The prefill plan performs embedding, repeated Transformer layers, final normalization, LM head projection, and sampling. Each layer includes RMSNorm, QKV projection, RoPE, causal GQA attention, output projection, residual updates, RMSNorm, gate/up projections, SwiGLU, down projection, and residual update.

- F16/BF16 linear algebra uses cuBLASLt as the initial baseline.
- Q4_K_M uses a custom or imported quantized GEMM implementation.
- Attention, RMSNorm, RoPE, and SwiGLU use custom fused kernels when validated.
- RAFT supplies selected reduction, argmax, selection, and RNG baselines.

Prefill accepts dynamic sequence lengths and is not required to run inside a CUDA Graph in v0.1.

### 7.3 KV cache

v0.1 uses a contiguous, fixed-address KV cache:

```text
[layer][K/V][token][kv_head][head_dim]
```

The session allocates it once for the configured maximum context length. It uses F16/BF16 values and a device-side token position. Reset changes logical length and state but does not reallocate memory.

### 7.4 Decode and CUDA Graph

Ordinary decode and graph decode use the same execution plan and kernel bindings. Once ordinary execution is correct, the engine captures the fixed-address, batch-one, single-token path. Decode-time allocation is prohibited.

A decode operation updates the input token, runs the Transformer, writes KV entries, computes logits, samples the next token, updates session counters, and returns a small result.

If capture fails, ordinary decode remains available and the engine reports the reason. A CUDA execution failure poisons the session until it is reset or recreated.

### 7.5 Sampling

- Greedy sampling starts with RAFT argmax or a custom fused argmax.
- Temperature, top-k, and top-p use RAFT selection and RNG primitives as the correctness baseline.
- A fused sampling kernel may replace the baseline only after the same correctness and performance gates.
- Session RNG seed and counter are explicit state.
- Ordinary and graph execution must be reproducible with the same seed within the project.

## 8. `bw24` reuse policy

The project is reuse-first for proven performance assets. It does not rewrite a `bw24` kernel merely to claim original implementation or to force RAFT into a hot path.

Candidate imports include optimized Q4_K/Q8_0/NVFP4 linear kernels, fused attention, fused normalization/activation, MoE routing for later versions, sampling, CUDA Graph device kernels, quantization layouts, and independent test/oracle material.

The import process is:

1. Identify the upstream file and pin its exact commit.
2. Confirm that its license and any transitive code are compatible.
3. Preserve the kernel algorithm, warp mapping, reduction order, and layout unless a measured need requires changes.
4. Adapt the host interface to `ExecutionContext`, the operator registry, and preallocated workspaces.
5. Run all project correctness, algorithmic, token-parity, graph, and performance gates.
6. Publish the provenance and target-GPU results.

Imported kernel code lives under `cpp/kernels/imported/bw24/` with a local README, upstream commit record, license text, modification summary, verified GPU list, and verification evidence.

`bw24` performance results are not inherited. Its principal tuning target is an RTX 5090 Laptop; desktop 5090, 5080, 5070 Ti, and other RTX 50 GPUs require fresh validation and may require different launch parameters or tiles.

Model files, benchmark assets, and third-party code referenced by `bw24` are independently licensed and are not assumed to be covered by the repository's MIT license.

## 9. Correctness contract

Correctness and performance are equal release gates. Performance data is invalid if an earlier correctness gate fails.

```text
format/layout validation
  → kernel numerical validation
  → algorithmic invariant validation
  → end-to-end token parity
  → performance and regression validation
```

### 9.1 Format and layout

Randomly sampled Q4_K_M blocks are decoded by an independent CPU implementation. Tests verify packed values, scale/min interpretation, dequantized results, tensor offsets, alignment, bounds, and any repacking transformation.

The CPU reference must not share the CUDA kernel's unpacking implementation.

### 9.2 Kernel numerical validation

Each kernel is tested using deterministic fixtures, randomized inputs, and adversarial values and shapes. Validation uses operator-specific tolerances and invariants rather than one global epsilon.

Representative comparisons include:

- RMSNorm and RoPE against CPU FP32.
- BF16 linear against CPU FP32 and cuBLASLt.
- Q4_K_M linear against CPU dequantization plus FP32 GEMM.
- Prefill and decode attention against CPU FP32.
- Argmax index equality against CPU.
- Sampling determinism and distribution checks using fixed RNG streams.

Metrics include maximum absolute error, relative error, cosine similarity, softmax row sums, and exact indices where appropriate.

### 9.3 Algorithmic invariants

Tests explicitly verify RMSNorm epsilon semantics, RoPE position/frequency/dimension ordering, causal masking, GQA head mapping, KV layer and token positions, Q4_K_M scaling semantics, graph token-counter updates, and sampling seed/counter behavior.

### 9.4 End-to-end parity

A fixed prompt corpus runs through llama.cpp, ordinary project execution, and CUDA Graph execution.

Greedy mode requires identical prompt token IDs, identical next-token argmax at every step, identical final token sequences, and identical project results between ordinary and graph paths. Diagnostics record the logits maximum error and top-two margin at every step.

A mismatch must be localized to the first divergent layer/operator. Final-text similarity is not an acceptance criterion.

Sampling mode must be reproducible inside the project for a fixed seed. It is not required to reproduce llama.cpp's random sequence if the RNG algorithms differ.

## 10. Performance contract

Kernel benchmarks run only after correctness passes. They record latency, effective bandwidth, effective throughput, workspace, shared memory, registers, occupancy, and launch count.

Required coverage includes:

- Prefill lengths 16, 128, 512, and 2048.
- Single-token decode.
- Context lengths 32, 512, 2048, and 8192.

A kernel becomes the default only when it passes all correctness and graph tests, provides a stable benefit over its fallback, and introduces no unexplained severe regression in an important shape bucket.

End-to-end v0.1 targets on the same hardware, model, GGUF file, and prompts are:

| Metric | Gate |
|---|---:|
| Model load | Recorded; no competitive gate |
| PP512 | At least 95% of llama.cpp |
| PP2048 | At least 95% of llama.cpp |
| TG128 decode | No lower than llama.cpp |
| One published scenario | At least 10% faster than llama.cpp |
| Greedy correctness | Identical token sequence |
| Ordinary/graph execution | Identical project token sequence |

Comparisons use at least five runs, medians, interleaved engines in the same session, controlled power/clock policy, and recorded temperature, frequency, GPU, driver, CUDA, model hash, prompt hash, kernel selection, and commit.

## 11. CI and evidence

Regular CI runs C++ and Rust builds, formatting, static analysis, CPU reference tests, GGUF fixtures, ABI checks, and mock execution tests.

RTX 50 local or self-hosted CI runs kernel correctness, CUDA Graph tests, Qwen3.5-9B token parity, kernel benchmarks, and interleaved llama.cpp comparisons.

GPU evidence is stored as machine-readable JSONL. Every published performance result must be traceable to the code commit, model hash, prompt hash, GPU, driver, CUDA version, selected kernels, correctness result, and measurement protocol.

### 11.1 Target validation host and shared-GPU policy

The initial remote validation host is `192.168.124.8`, accessed as user `charles`. Authentication secrets are never written to the repository, build scripts, logs, command history, benchmark evidence, or documentation. Tests use an interactive credential prompt or an approved secret mechanism.

The target GPU may be shared with unrelated workloads. Remote test automation must therefore follow these rules:

- Inspect GPU processes, memory use, utilization, temperature, clocks, and power state before allocating significant memory or starting a benchmark.
- Never terminate, suspend, renice, or otherwise interfere with another process unless the user explicitly authorizes that exact action.
- Never enable exclusive-process mode, reset the GPU, change persistent system GPU configuration, or change global power/clock settings without explicit authorization.
- Correctness tests may run alongside other processes only when the required memory headroom is available and the test cannot cause an out-of-memory condition for existing workloads.
- Performance tests require a quiet measurement window. They must not start when unrelated GPU compute utilization is active or when free VRAM is below the scenario's declared safety margin.
- Re-check occupancy before every benchmark arm. If a competing workload appears, abort the current measurement cleanly and mark its data invalid rather than publishing a distorted result.
- Use a cooperative host-local benchmark lock to prevent two project benchmark jobs from overlapping. The lock does not grant ownership of the GPU and does not override evidence of unrelated workloads.
- Record pre-run and post-run GPU state in benchmark evidence. A result is publishable only when the evidence shows a stable, uncontended measurement window.
- Prefer small smoke and correctness probes before full-model or long-running performance tests.

Remote tests must stage outputs in a project-specific directory, avoid modifying unrelated files or services, and clean up only artifacts created by the current project run.

## 12. Milestones

### M0: Toolchain and full-stack smoke

- CMake builds C++20/CUDA with RAFT, RMM, CUTLASS, and cuBLASLt.
- Cargo builds the Rust system, runtime, and CLI crates.
- The stable C ABI creates an engine.
- A Rust CLI prints GPU capabilities, allocates RMM memory, launches a smoke kernel, and reads the correct result.
- Exact tested dependency revisions and the `sm_120a` compilation mode are locked.
- The same smoke test runs safely on the target validation host after passing the shared-GPU preflight checks.

### M1: Foundation, registry, and correctness harness

- Implement tensor descriptions, execution context, workspace arena, registry, status model, reference framework, fixtures, randomized tests, and benchmark JSONL.
- Register reference RMSNorm, RoPE, BF16 linear, softmax/argmax, embedding, Add, and SwiGLU operators.
- Demonstrate dispatch, fallback, and graph-safety metadata.

### M2: Qwen3.5-9B F16/BF16 baseline

- Implement GGUF loading, Rust tokenization, Qwen3.5 configuration, full Dense forward, prefill, contiguous KV cache, ordinary decode, greedy sampling, and CLI generation.
- Match llama.cpp greedy tokens on the fixed corpus.
- Record the first end-to-end performance baseline.

### M3: Q4_K_M and `bw24` kernel reuse

- Inventory candidate upstream kernels as direct import, thin adaptation, rewrite, or unsuitable.
- Import and adapt applicable Q4_K_M and fused kernels with complete provenance.
- Validate Q4_K_M layout and CPU dequantization.
- Match llama.cpp greedy tokens on Qwen3.5-9B Q4_K_M.

### M4: Optimized attention and CUDA Graph

- Add validated custom prefill and decode attention.
- Capture the fixed-address decode plan with a device token position.
- Demonstrate identical project tokens between ordinary and graph execution.
- Eliminate decode-time allocation and cache kernel selections.

### M5: v0.1 hardening and release

- Pass all correctness and performance gates.
- Publish Qwen3.5-9B F16/BF16 and Q4_K_M support on RTX 50/sm_120a.
- Publish reproducible benchmark evidence, architecture documentation, build instructions, provenance records, and contribution requirements.

## 13. Licensing

The project should use Apache License 2.0 for its original code. This aligns with RAFT/RMM and provides explicit patent terms. MIT-licensed `bw24` code may be incorporated while retaining its copyright and license notices.

The repository maintains:

- `LICENSE` for project-original code.
- `NOTICE` for required notices.
- `THIRD_PARTY_LICENSES/` for dependency and imported-code licenses.
- `docs/provenance/` for imported files, commits, modifications, and validation evidence.

No model, benchmark asset, or third-party source is redistributed until its separate license has been reviewed.

## 14. Release definition

v0.1 is complete only when:

- Qwen3.5-9B Dense GGUF runs end to end with F16/BF16 and Q4_K_M.
- Rust drives the C++/CUDA engine through the stable ABI.
- RAFT/RMM form the common GPU resource and memory foundation.
- Imported and native optimized kernels use the common registry and execution context.
- All required numerical, algorithmic, token-parity, graph, and error-path tests pass.
- Decode matches or exceeds llama.cpp and one published scenario is at least 10% faster.
- Every imported kernel and benchmark result is traceable and license-compliant.
- The published documentation accurately states the RTX 50-only scope and all remaining limitations.
