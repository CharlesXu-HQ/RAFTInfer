# M2A GGUF and Qwen3.5 Loader Implementation Plan

> Execute task-by-task with tests written before production behavior.

**Goal:** Load and validate a GGUF v3 catalog, derive an immutable Qwen3.5 text
configuration and ordered hybrid block plan, and expose this functionality as
the first M2 model-loading slice.

**Architecture:** A dependency-free, bounds-checked binary reader produces a
typed metadata/tensor catalog. A separate Qwen3.5 adapter consumes that catalog
and validates architecture fields without coupling generic GGUF parsing to
model semantics. C API model handles are added only after both internal layers
are independently tested.

**Tech stack:** C++20, CMake/CTest, existing status/C ABI framework.

---

## Task 1: Generic GGUF value and catalog types

**Files**

- Create: `cpp/model/gguf_types.hpp`
- Test: `cpp/tests/gguf_reader_test.cpp`
- Modify: `cpp/CMakeLists.txt`

Write compile-time and behavioral tests for scalar metadata access, array
access, tensor lookup, duplicate rejection helpers, and public limits. Add the
minimal value/catalog representation needed to make them pass.

Commit: `feat: add typed GGUF catalog`

## Task 2: Bounds-checked GGUF v3 reader

**Files**

- Create: `cpp/model/gguf_reader.hpp`
- Create: `cpp/model/gguf_reader.cpp`
- Extend: `cpp/tests/gguf_reader_test.cpp`
- Modify: `cpp/CMakeLists.txt`

Build deterministic GGUF bytes in the test without checked-in model data.
Cover valid scalar/array metadata and tensor descriptors first. Then cover bad
magic, unsupported version, truncation at every field category, invalid types,
excessive lengths/counts, duplicate names, invalid alignment, arithmetic
overflow, out-of-file tensor spans, and overlap.

Run:

```bash
cmake -S . -B build-m2a -DRAFTINFER_ENABLE_CUDA=OFF -DRAFTINFER_BUILD_TESTS=ON
cmake --build build-m2a --target raftinfer_gguf_reader_test
ctest --test-dir build-m2a -R raftinfer_gguf_reader_test --output-on-failure
```

Commit: `feat: parse and validate GGUF v3 catalogs`

## Task 3: Qwen3.5 text configuration and block plan

**Files**

- Create: `cpp/model/qwen35_config.hpp`
- Create: `cpp/model/qwen35_config.cpp`
- Create: `cpp/tests/qwen35_config_test.cpp`
- Modify: `cpp/CMakeLists.txt`

Start with a minimal synthetic metadata catalog representing the official
32-block configuration. Require architecture, dimensions, head counts,
convolution width, normalization epsilon, RoPE fields, and ordered layer types.
Validate cross-field divisibility and tensor-independent invariants. Reject
conventional-Qwen3 metadata and any unknown block type.

Run:

```bash
cmake --build build-m2a --target raftinfer_qwen35_config_test
ctest --test-dir build-m2a -R raftinfer_qwen35_config_test --output-on-failure
```

Commit: `feat: derive Qwen3.5 hybrid block plans`

## Task 4: Tensor manifest validation

**Files**

- Create: `cpp/model/qwen35_manifest.hpp`
- Create: `cpp/model/qwen35_manifest.cpp`
- Create: `cpp/tests/qwen35_manifest_test.cpp`
- Modify: `cpp/CMakeLists.txt`

Define semantic tensor roles for embeddings, norms, MLPs, full attention, and
Gated DeltaNet. Validate required presence, rank, dimensions, layer ownership,
and that known sidecar tensors cannot substitute for text tensors. Do not
validate quantized payload blocks in M2A.

Commit: `feat: validate Qwen3.5 text tensor manifests`

## Task 5: Model handle and tokenizer-spec plumbing

**Files**

- Modify: `cpp/include/raftinfer/c_api.h`
- Modify: `cpp/src/c_api.cpp`
- Modify: `cpp/src/engine.hpp`
- Modify: `cpp/src/engine.cpp`
- Create: `cpp/model/model.hpp`
- Create: `cpp/model/model.cpp`
- Extend: `cpp/tests/c_api_test.cpp`
- Extend: `cpp/tests/c_api_source_test.cmake`
- Modify: `rust/raftinfer-sys/src/lib.rs`
- Extend: `rust/raftinfer-runtime/tests/engine.rs`

Add opaque model ownership and atomic load failure. Initially the model owns the
mapped catalog/config/manifest, not GPU weights. Add an owned, versioned
tokenizer-spec buffer with explicit allocation/free. Ensure destruction order
is safe when Rust drops wrappers.

Commit: `feat: expose validated GGUF models through coarse ABI`

## Task 6: Full local verification and evidence

**Files**

- Modify: `README.md`
- Create: `docs/verification/m2a.md`

Run fresh Debug and Release host builds, all CTest tests, Cargo tests, formatting
checks, and the existing local check script. Record exact commands, results,
scope, and remaining GPU gap. M2A has no CUDA behavior and does not require the
shared GPU host.

Commit: `docs: record M2A loader verification`

