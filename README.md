# RAFTInfer

RAFTInfer is an open-source Qwen3.5 inference runtime for
RTX 50-series GPUs. RAFT and RMM provide the device-resource, stream, and memory
foundation; performance-critical model operators are implemented in project
C++/CUDA code and selected through stable execution plans.

Current status: M2 is **accepted** for the Qwen3.5-9B text path on RTX 50. The
CUDA build and all 23 target CTest tests pass, four real-model prompts match a
pinned llama.cpp reference exactly under greedy decoding, and both PP128/TG128
and PP512/TG128 exceed the required 0.8 throughput ratio. Exact evidence and
reproduction paths are recorded in
[docs/verification/m2.md](docs/verification/m2.md).

## Scope

- RTX 50-series consumer Blackwell only (`sm_120a`). RTX 40 is not supported.
- Qwen3.5-9B text generation in BF16/F16. Vision, MTP, and general quantized
  execution are deferred.
- C++ owns GPU pointers, streams, events, RMM allocations, persistent model
  state, cuBLASLt plans, and custom kernel launches.
- Rust owns the safe public runtime, tokenizer/chat template, generation loop,
  machine-readable CLI output, and benchmark orchestration.
- Rust crosses the coarse C ABI at model/session operations such as prefill,
  decode, reset, and logits copy; it does not call individual CUDA operators.
- A `bw24` custom operator may be reused directly when it is already optimized
  and passes license, provenance, functional/numerical correctness, and RTX 50
  performance gates. The current M2 CUDA operators are project-native because
  the audited `bw24` material contained no reusable optimized implementation.

## What M2 contains

- Bounds-checked GGUF v3 catalog, Qwen3.5 hybrid block-plan validation, named
  immutable CUDA weights, and fixed cuBLASLt projection plans.
- Independent CPU FP32 reference semantics for the full hybrid executor.
- FP32-activation CUDA primitives with BF16/F16 projection boundaries for
  RMSNorm, RoPE/full attention, Gated DeltaNet, gated MLP, residual flow,
  prefill, decode, and persistent session state.
- RMM-backed fixed workspace/state allocation; execution loops are tested to
  perform no additional RMM allocation.
- Rust Qwen3.5 tokenizer/chat-template handling and greedy generation.
- JSON generation output carrying both prompt and generated token IDs.
- One-process prefill/decode benchmark support.
- RMM logical-allocation peak tracking exposed through the stable C ABI and
  included in benchmark JSON.
- Reproducible GGUF preparation, fixed-corpus llama.cpp parity, and fair
  PP128/PP512 + TG128 benchmark scripts.

## Host verification

```bash
scripts/local-check.sh
cargo clippy --workspace --all-targets -- -D warnings
```

`local-check.sh` builds the host-only C++ implementation, runs CTest, verifies
native static/shared library selection, exercises the parity/benchmark/GGUF
scripts with controlled fakes, checks Rust formatting, and runs all Rust tests.

## CUDA build and CLI

The CUDA build requires CUDA 13, RAFT 26.06, and RMM 26.06. The repository's
development image supplies this environment.

```bash
cmake -S . -B build/cuda -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=ON \
  -DRAFTINFER_BUILD_TESTS=ON
cmake --build build/cuda
scripts/gpu-preflight.sh
ctest --test-dir build/cuda --output-on-failure

RAFTINFER_ENABLE_CUDA=ON cargo build --release -p raftinfer-cli
target/release/raftinfer generate \
  --model /path/to/Qwen3.5-9B-bf16.gguf \
  --prompt "用一句话解释 RAFT 在本项目中的作用。" \
  --max-new-tokens 32 \
  --context 4096 \
  --output-format json
```

CUDA builds default `raftinfer_cpp` to a shared native library so the CMake target
encapsulates the CUDA/RAFT/RMM/cuBLASLt dependency closure. Host builds default
to a static library. `RAFTINFER_NATIVE_LIBRARY_TYPE=STATIC|SHARED` can override this
choice explicitly.

## Parity and performance

Prepare a new artifact only from pinned model, Transformers, converter, and
reference revisions:

```bash
HF_MODEL_DIR=/path/to/Qwen3.5-9B \
HF_MODEL_REVISION=<40-char-revision> \
TRANSFORMERS_REVISION=<40-char-revision> \
LLAMA_CONVERTER_DIR=/path/to/llama.cpp-converter \
LLAMA_CONVERTER_REVISION=<40-char-revision> \
LLAMA_REFERENCE_DIR=/path/to/llama.cpp-reference \
LLAMA_REFERENCE_REVISION=<40-char-revision> \
OUTPUT_GGUF=/path/to/Qwen3.5-9B-bf16.gguf \
PROVENANCE_OUTPUT=/path/to/Qwen3.5-9B-bf16.provenance.json \
scripts/prepare-qwen35-gguf.sh
```

Then run exact token parity before benchmarking:

```bash
RAFTINFER_MODEL=/path/to/Qwen3.5-9B-bf16.gguf \
LLAMA_SERVER_BIN=/path/to/llama-server \
scripts/qwen35-parity.sh

RAFTINFER_MODEL=/path/to/Qwen3.5-9B-bf16.gguf \
LLAMA_SERVER_BIN=/path/to/llama-server \
scripts/qwen35-benchmark.sh
```

Both GPU scripts take a cooperative lock and execute
`scripts/gpu-preflight.sh`. The benchmark refuses to start unless every parity
record passes. On a shared GPU, do not bypass this guard or stop unrelated
processes.

See [docs/verification/m2.md](docs/verification/m2.md) for the current evidence,
[docs/provenance/qwen35-9b.md](docs/provenance/qwen35-9b.md) for artifact
provenance status, [docs/verification/m1.md](docs/verification/m1.md) for M1,
and [docs/provenance/dependencies.md](docs/provenance/dependencies.md) for
dependency provenance.
