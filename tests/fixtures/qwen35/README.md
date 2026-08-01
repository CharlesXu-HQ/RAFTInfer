# Qwen3.5 tiny layer fixture

`tiny-layer-v1.bin` is deterministic CPU FP32 truth generated from the
official Hugging Face Transformers Qwen3.5 implementation. It does not contain
any Qwen3.5-9B checkpoint weight.

Pinned source:

- Transformers version: `5.14.1`
- Release commit:
  `a08ace4bbd97e721c98751deec37d87b026acadc`
- `modeling_qwen3_5.py` SHA-256:
  `0e2cd8dc50885b2701d26b116c585eedcdc62a24080ec34345af55b963126ded`
- Generator seed: `20260725`
- Fixture SHA-256:
  `800878611016996887aa9828ee97eb165e6bf1066543a8797b4c99e6eaf167e0`

Generate with GPU access disabled:

```sh
CUDA_VISIBLE_DEVICES="" python tools/qwen35/export_reference_fixtures.py \
  --output tests/fixtures/qwen35/tiny-layer-v1.bin
```

The version-1 little-endian file begins with:

1. magic `RIFQ35F1`
2. `uint32` version
3. `uint32` endian marker `0x01020304`
4. `uint64` exact total file length
5. fixed full-attention and Gated DeltaNet configuration fields
6. fixed-order vectors, each encoded as `uint64 count` followed by FP32 data

Vector order:

1. canonical full-attention `[q,k,v,gate]` input
2. full-attention Q norm weight
3. full-attention K norm weight
4. full-attention output weight in `[input,output]` order
5. official full-attention output
6. canonical DeltaNet `[q,k,v,b,a,z]` input
7. depthwise convolution weight in `[channel,kernel]` order
8. GGUF-native recurrent coefficient `-exp(A_log)`
9. `dt_bias`
10. direct output RMSNorm weight
11. initial convolution state `[channel,history]`
12. initial recurrent state `[value_head,key_dim,value_dim]`
13. official DeltaNet output
14. final convolution state
15. final recurrent state

The exporter uses the official eager Qwen3.5 attention module and the official
PyTorch causal-convolution, recurrent Gated DeltaNet, chunk Gated DeltaNet, and
gated RMSNorm implementations. It asserts that chunk and recurrent results
agree before writing the fixture. The C++ test independently recomputes the
same fields through `cpp/reference/qwen35.cpp` at `1e-5` absolute or relative
tolerance and rejects bad magic, version, truncation, trailing data, or a
declared-length mismatch.
