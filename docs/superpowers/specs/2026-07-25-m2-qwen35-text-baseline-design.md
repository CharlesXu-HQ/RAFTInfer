# M2 Qwen3.5-9B Text Baseline Design

Date: 2026-07-25

Status: Approved amendment to the 2026-07-22 engine design

## 1. Decision

M2 targets the text-generation path of `Qwen/Qwen3.5-9B`.

Qwen3.5-9B is not a conventional Dense Transformer. Its text decoder contains
32 dense-MLP blocks with a 3:1 token-mixer pattern:

- 24 Gated DeltaNet linear-attention blocks;
- 8 causal full-attention blocks, one after every three linear blocks.

The model is dense in the MoE sense, but its state and execution plan are
hybrid. M2 must therefore implement both token mixers. Treating every block as
full attention would be functionally incorrect.

The official model is multimodal, but M2 intentionally implements text-only
generation. Vision encoding, image/video tokens, the multimodal projector, and
the MTP head are deferred. The loader accepts a text-model GGUF and rejects an
mmproj/MTP sidecar as the primary model.

## 2. Source of truth

Architecture semantics are pinned to:

- the official Qwen3.5-9B `config.json`;
- the matching Hugging Face Transformers Qwen3.5 implementation;
- the GGUF v3 specification and a pinned llama.cpp converter/reference build.

At the time of this decision the official Qwen repository publishes
Safetensors, tokenizer assets, and configuration, but no official GGUF. The
acceptance GGUF is generated from the official checkpoint with a pinned
llama.cpp conversion revision. The project remains GGUF-only at runtime and
does not add a Safetensors loader.

Relevant upstream references:

- https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

Every acceptance artifact records the upstream model revision, converter
revision, conversion command, GGUF SHA-256, and llama.cpp reference revision.

## 3. Verified Qwen3.5-9B text configuration

The loader does not silently infer missing architecture values. It extracts the
GGUF metadata and validates the resulting configuration against tensor shapes.
The official 9B configuration currently has:

| Field | Value |
|---|---:|
| vocabulary | 248320 |
| hidden size | 4096 |
| intermediate size | 12288 |
| blocks | 32 |
| full-attention interval | 4 |
| full-attention query heads | 16 |
| full-attention KV heads | 4 |
| full-attention head dimension | 256 |
| linear key heads | 16 |
| linear value heads | 32 |
| linear key/value head dimension | 128 |
| causal convolution width | 4 |
| RMSNorm epsilon | 1e-6 |
| maximum trained positions | 262144 |
| RoPE theta | 10000000 |
| partial rotary factor | 0.25 |

The exact ordered block-type array remains authoritative. The interval is a
consistency check, not a replacement for the array when the GGUF contains it.

## 4. M2 decomposition

M2 is delivered as four independently verifiable vertical slices.

### M2A: GGUF catalog and model configuration

- Bounds-checked GGUF v3 reader.
- Typed metadata values and tensor catalog.
- Checked arithmetic for dimensions, offsets, alignment, and byte spans.
- Qwen3.5 text configuration extraction.
- Ordered block-plan construction.
- Tokenizer metadata extraction into a versioned internal specification.
- Synthetic, small, redistributable GGUF fixtures and corruption tests.

M2A maps only the file header and tensor catalog. It does not upload weights.

### M2B: Hybrid reference semantics and session state

- Independent CPU FP32 Gated DeltaNet reference.
- Qwen3.5 RMSNorm `1 + weight` semantics.
- Gated full attention with Q/K normalization and partial RoPE.
- Hybrid session state layout and reset semantics.
- Layer-level reference fixtures captured from the official implementation.

### M2C: CUDA BF16/F16 baseline

- Immutable GPU weight plan and RMM-backed allocations.
- cuBLASLt baselines for projections.
- Correct prefill for both block types.
- Correct single-token recurrent update for linear blocks.
- Correct causal attention and KV update for full-attention blocks.
- Greedy logits/argmax path.

Optimized `bw24` kernels are reused directly when they already implement an
applicable operation and pass provenance, license, functional, numerical, and
performance gates. General Q4_K_M integration remains M3.

### M2D: Rust tokenizer, CLI, and parity

- Versioned tokenizer-spec buffer through the coarse C ABI.
- Rust tokenizer/chat-template implementation.
- Model/session RAII wrappers.
- Prefill and decode calls at request-step granularity.
- Fixed-corpus prompt-token and greedy-token parity with llama.cpp.
- First end-to-end correctness and performance evidence.

## 5. GGUF reader contract

The reader is a small project-owned implementation with no new dependency. It:

- accepts GGUF version 3 only for M2;
- requires little-endian encoding;
- enforces explicit limits for metadata count, tensor count, string length,
  array length, tensor rank, and total catalog memory;
- rejects duplicate metadata keys and tensor names;
- supports every scalar metadata type and homogeneous, non-nested arrays;
- uses checked addition and multiplication before every allocation or span
  calculation;
- validates `general.alignment` as a nonzero multiple of eight;
- validates every tensor data span against file size;
- rejects overlapping tensor spans;
- reports the metadata key or tensor name in structured error details.

Arrays of arrays are rejected in M2 even though the extensible format can
represent them; Qwen3.5 model/tokenizer metadata does not require nesting.

Tensor payloads are not decoded by the catalog reader. Data-type-specific byte
size and block-layout validation lives in the weight-format module introduced
with GPU upload or quantization support.

## 6. Hybrid execution and state

An immutable block plan contains one entry per decoder block:

```text
BlockPlan {
  block_index,
  mixer = GatedDeltaNet | FullAttention,
  tensor bindings,
  prefill operator bindings,
  decode operator bindings
}
```

The session owns two kinds of state:

```text
Full-attention layers:
  [full_layer][K/V][token][kv_head][head_dim]

Linear-attention layers:
  convolution_state[linear_layer][qkv_channel][conv_width - 1]
  recurrent_state[linear_layer][value_head][key_dim][value_dim]
```

The recurrent state is FP32 for the baseline, matching the official numerical
semantics. KV and convolution state use the configured activation dtype unless
reference evidence requires a wider type.

Reset zeroes logical position, convolution state, recurrent state, and KV
length without reallocating. Decode-time allocation remains prohibited.

## 7. Correctness gates

M2 uses layered correctness:

1. GGUF structural and model-configuration validation.
2. Operator comparison against independent CPU FP32 references.
3. Per-block hidden-state comparison against fixtures from the official model.
4. Hybrid-state transition checks for prefill, continued prefill, decode, and
   reset.
5. Prompt tokenizer equality with llama.cpp.
6. Exact greedy next-token equality at every step on the fixed corpus.

Required special cases include:

- Qwen3.5 RMSNorm weights are applied as `1 + weight`;
- full attention applies Q/K normalization and the output sigmoid gate;
- only the configured fraction of each head receives RoPE;
- linear attention maintains both convolution and recurrent state;
- the ordered 3-linear/1-full block plan is preserved;
- full-attention KV positions and linear recurrent updates advance atomically.

Performance measurements are invalid until the corresponding correctness gates
pass.

## 8. ABI impact

The public ABI grows only at model/session granularity:

```c
BrtStatus brt_engine_load_model(
    BrtEngineHandle*, const char* gguf_path, BrtModelHandle**);
BrtStatus brt_model_copy_tokenizer_spec(
    const BrtModelHandle*, BrtOwnedBuffer*);
BrtStatus brt_session_create(
    BrtModelHandle*, const BrtSessionConfig*, BrtSessionHandle**);
BrtStatus brt_session_prefill(
    BrtSessionHandle*, const int32_t*, size_t, BrtPrefillResult*);
BrtStatus brt_session_decode(
    BrtSessionHandle*, const BrtDecodeParams*, BrtDecodeResult*);
```

Rust never submits individual blocks or operators. C++ builds the hybrid plan
once and executes one complete prefill or decode step per FFI call.

## 9. Acceptance and deferred scope

M2 completes when a converted Qwen3.5-9B text GGUF runs end to end in BF16/F16
on the RTX 50 target and produces exact llama.cpp greedy tokens on the fixed
corpus.

Deferred:

- vision tower and multimodal projector;
- image/video preprocessing and multimodal RoPE inputs;
- MTP/speculative decoding;
- Q4_K_M and other quantized formats;
- CUDA Graph capture;
- paged or quantized state caches;
- multi-session scheduling.

