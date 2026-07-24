# M2A Qwen3.5 Loader Verification

## Scope

M2A is the host-side loading foundation for the text path of Qwen3.5-9B. It
contains:

- a bounds-checked GGUF v3 metadata and tensor-catalog reader;
- extraction and validation of the Qwen3.5 hybrid block configuration;
- exact tensor-role and shape validation for linear-attention and
  full-attention blocks;
- read-only file mapping owned by an opaque C++ model handle;
- a versioned, owned tokenizer-metadata buffer copied through the coarse C ABI;
- Rust model lifetime and error wrappers.

The model loader publishes no handle until the file, configuration, and tensor
manifest have all passed validation. The Rust `Model` lifetime is tied to its
`Engine`, and both native model and tokenizer buffers use RAII-style
destruction.

## Fresh host verification

Verified on 2026-07-25 with AppleClang 21.0.0.21000101 and CUDA disabled.

Debug:

```bash
cmake -S . -B build/m2a-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBRT_ENABLE_CUDA=OFF \
  -DBRT_BUILD_TESTS=ON
cmake --build build/m2a-debug -j4
ctest --test-dir build/m2a-debug --output-on-failure
```

Result: 12/12 CTest tests passed.

Release:

```bash
cmake -S . -B build/m2a-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBRT_ENABLE_CUDA=OFF \
  -DBRT_BUILD_TESTS=ON
cmake --build build/m2a-release -j4
ctest --test-dir build/m2a-release --output-on-failure
```

Result: 12/12 CTest tests passed. Assert-based tests remain active in Release.

Repository check:

```bash
scripts/local-check.sh
```

Result:

- 12/12 CTest tests passed;
- 2/2 CLI integration tests passed;
- 4/4 Rust runtime integration tests passed;
- Rust unit and documentation tests passed;
- formatting checks passed.

The success-path fixture is a small synthetic four-block Qwen3.5 GGUF. It
exercises the required three Gated DeltaNet blocks followed by one
full-attention block, all common MLP/norm tensors, tokenizer vocabulary and
merges, the chat template, model-handle creation, and owned-buffer release.
Corruption, missing-field, invalid-shape, and missing-file paths are covered by
the GGUF, configuration, manifest, C ABI, and Rust tests.

## Deferred validation

M2A performs no CUDA work, so the shared RTX 50 target was intentionally not
used. The following remain later M2 slices:

- loading a real converted Qwen3.5-9B GGUF fixture;
- weight upload and immutable GPU weight plans;
- CPU reference semantics for Gated DeltaNet and gated full attention;
- BF16/F16 CUDA prefill and decode;
- Rust tokenizer interpretation and llama.cpp token parity;
- end-to-end greedy-token parity and performance measurement.

Optimized `bw24` kernels remain eligible for direct reuse when their operation
matches Qwen3.5 and they pass provenance, license, functional, numerical, and
RTX 50 performance gates.
