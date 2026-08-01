# RAFTInfer Open-Source Renaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely rename the project to RAFTInfer, make the repository publication-ready with synchronized English and Chinese documentation plus reproducible benchmark visuals, verify the renamed runtime on the RTX 5090, and publish the verified `main` branch to `charlesxu91/RAFTInfer`.

**Architecture:** Preserve the existing top-level C++/CUDA and Rust separation while replacing every active technical identifier at the C ABI, CMake, Rust, script, schema, and documentation boundaries. Add a repository-level brand contract before the mechanical rename, then update one dependency layer at a time. Treat committed benchmark JSONL as the source of truth for a deterministic SVG and README claims.

**Tech Stack:** C++20, CUDA 13.2, CMake, Ninja, RAFT 26.06, RMM 26.06, cuBLASLt, Rust 2024, Bash, jq, Python 3 standard library, GitHub Actions.

## Global Constraints

- The public project name is exactly `RAFTInfer`.
- The C++ namespace and public include root are exactly `raftinfer`.
- C ABI types use the `RaftInfer` prefix and C ABI functions use the `raftinfer_` prefix.
- Rust packages are `raftinfer-sys`, `raftinfer-runtime`, and `raftinfer-cli`; the executable is `raftinfer`.
- CMake options and runtime environment variables use the `RAFTINFER_` prefix.
- No old-name compatibility headers, symbols, packages, environment fallbacks, or wrappers are retained.
- Old identifiers may appear only in the approved migration design and historical entries in `CHANGELOG.md`.
- RTX 50-series Blackwell (`sm_120a`) remains the only supported GPU family.
- README performance charts compare RAFTInfer with llama.cpp only.
- Published benchmark claims use the pinned Qwen3.5-9B BF16 artifact, llama.cpp revision `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`, 5 warmups, 20 measurements, and 4-by-32 exact parity.
- No new third-party runtime or documentation dependency is added.
- The GitHub destination is `https://github.com/charlesxu91/RAFTInfer.git`.
- Unknown remote history is never overwritten with force push.

---

### Task 1: Lock the repository brand and publication contract

**Files:**
- Create: `scripts/check-project-brand.sh`
- Create: `tests/public-surface-test.sh`
- Create: `tests/readme-links-test.py`
- Modify: `scripts/local-check.sh`
- Test: `tests/public-surface-test.sh`
- Lint: `scripts/check-project-brand.sh`
- Test: `tests/readme-links-test.py`

**Interfaces:**
- Consumes: the design allowlist for the migration specification and `CHANGELOG.md`.
- Produces: a failing behavioral test for the renamed public surface, a
  separate static repository-brand policy check, and a dependency-free local
  Markdown link validator.

- [ ] **Step 1: Add the failing public-surface behavior test**

Create an executable test that:

1. configures and builds the host CMake project using only
   `RAFTINFER_ENABLE_CUDA`, `RAFTINFER_BUILD_TESTS`, and the
   `raftinfer_cpp` target;
2. compiles and links a small C consumer that includes
   `<raftinfer/c_api.h>`, constructs `RaftInferEngineConfig`, and calls
   `raftinfer_engine_create` and `raftinfer_engine_destroy`;
3. reads `cargo metadata --format-version 1 --no-deps` and requires exactly
   `raftinfer-sys`, `raftinfer-runtime`, and `raftinfer-cli`;
4. runs the host `raftinfer info` command and requires it to report the host
   backend successfully.

This test verifies public behavior, not source spelling.

- [ ] **Step 2: Run the public-surface test and verify the expected red state**

Run:

```bash
tests/public-surface-test.sh
```

Expected: FAIL because the RAFTInfer CMake options, C header, packages, and CLI
do not exist before the rename.

- [ ] **Step 3: Add the static project-brand policy check**

Create an executable Bash linter that:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

required=(
  README.md
  README.zh-CN.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  SECURITY.md
  CHANGELOG.md
  CITATION.cff
  .github/CODEOWNERS
  .github/workflows/ci.yml
  benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl
  docs/assets/qwen35-bf16-rtx5090.svg
)
for path in "${required[@]}"; do
  test -f "${path}" || {
    printf 'project-brand: missing %s\n' "${path}" >&2
    exit 1
  }
done

legacy_pattern='Blackwell RAFT Runtime|namespace brt|include/brt|Brt[A-Z]|brt_[A-Za-z]|BRT_[A-Z]|brt-(sys|runtime|cli|dev)|"brt"|"brt_cpp"|\.brt([."]|$)|brt_model_sha256'
if git grep -I -nE "${legacy_pattern}" -- \
  . \
  ':!docs/superpowers/specs/2026-07-30-raftinfer-open-source-renaming-design.md' \
  ':!docs/superpowers/plans/2026-07-30-raftinfer-open-source-renaming.md' \
  ':!CHANGELOG.md'; then
  printf 'project-brand: active legacy identifier found\n' >&2
  exit 1
fi

git grep -q 'README.zh-CN.md' -- README.md
git grep -q 'README.md' -- README.zh-CN.md
git grep -q 'schema_version.*2' -- \
  tests benchmarks scripts cpp rust
```

- [ ] **Step 4: Add the local README link checker**

Implement `tests/readme-links-test.py` with only `pathlib`, `re`, and
`urllib.parse`. It reads both README files, ignores HTTP(S), anchors, badges,
and image data URLs, URL-decodes local targets, strips anchors, and fails when
the referenced repository-relative path does not exist.

- [ ] **Step 5: Register the behavioral test and both linters**

Add:

```bash
tests/public-surface-test.sh
scripts/check-project-brand.sh
python3 tests/readme-links-test.py
```

before Rust tests so repository contract failures are reported early.

- [ ] **Step 6: Run the checks and verify the complete red state**

Run:

```bash
tests/public-surface-test.sh
scripts/check-project-brand.sh
python3 tests/readme-links-test.py
```

Expected:

- `public-surface-test.sh` fails on the missing RAFTInfer build/API surface.
- `check-project-brand.sh` fails because RAFTInfer community files and renamed
  identifiers do not exist yet.
- `readme-links-test.py` passes against the current English README or reports
  only links that the implementation must repair.

- [ ] **Step 7: Commit the regression contract**

```bash
git add scripts/check-project-brand.sh tests/public-surface-test.sh \
  tests/readme-links-test.py scripts/local-check.sh
git commit -m "test: define RAFTInfer repository contract"
```

---

### Task 2: Rename the C++ API, CMake project, and native artifacts

**Files:**
- Move: `cmake/BrtOptions.cmake` → `cmake/RAFTInferOptions.cmake`
- Move: `cpp/include/brt/` → `cpp/include/raftinfer/`
- Modify: `CMakeLists.txt`
- Modify: `cpp/CMakeLists.txt`
- Modify: every tracked `cpp/**/*.cpp`, `cpp/**/*.cu`, `cpp/**/*.hpp`, `cpp/**/*.cuh`, and public header
- Modify: CMake source-inspection tests under `cpp/tests/`
- Test: all host CTest targets

**Interfaces:**
- Consumes: existing C++ behavior and public C ABI layouts.
- Produces: `raftinfer::` C++ APIs, `RaftInfer*` C types,
  `raftinfer_*` C functions, and the `raftinfer_cpp` native library.

- [ ] **Step 1: Move the CMake options file and public include directory**

Run:

```bash
git mv cmake/BrtOptions.cmake cmake/RAFTInferOptions.cmake
git mv cpp/include/brt cpp/include/raftinfer
```

- [ ] **Step 2: Rename CMake project contracts**

Update:

```cmake
project(raftinfer VERSION 0.1.0 LANGUAGES CXX)
include(cmake/RAFTInferOptions.cmake)
```

Rename:

- `BRT_ENABLE_CUDA` → `RAFTINFER_ENABLE_CUDA`
- `BRT_BUILD_TESTS` → `RAFTINFER_BUILD_TESTS`
- `BRT_NATIVE_LIBRARY_TYPE` → `RAFTINFER_NATIVE_LIBRARY_TYPE`
- `brt_cpp` → `raftinfer_cpp`
- `brt_reference` → `raftinfer_reference`
- `brt-smoke` → `raftinfer-smoke`
- every `brt_*_test` CTest target → `raftinfer_*_test`

The installed native filename becomes `libraftinfer_cpp.so` on Linux and
`libraftinfer_cpp.dylib` on macOS.

- [ ] **Step 3: Rename C++ namespaces and includes mechanically**

Apply bounded replacements only to tracked C++ and CMake files:

```text
namespace brt                 → namespace raftinfer
brt::                         → raftinfer::
<brt/                         → <raftinfer/
BRT_QWEN35_EXECUTOR_TESTING   → RAFTINFER_QWEN35_EXECUTOR_TESTING
BRT_TEST_CUDA_ENABLED         → RAFTINFER_TEST_CUDA_ENABLED
BRT_RUN_GPU_TESTS             → RAFTINFER_RUN_GPU_TESTS
```

Do not change `bw24` provenance text.

- [ ] **Step 4: Rename the complete C ABI**

In `cpp/include/raftinfer/status.h` and `cpp/include/raftinfer/c_api.h`, rename
all public items consistently:

```text
BrtStatusCode          → RaftInferStatusCode
BrtStatus              → RaftInferStatus
BrtEngineConfig        → RaftInferEngineConfig
BrtEngineHandle        → RaftInferEngineHandle
BrtModelHandle         → RaftInferModelHandle
BrtSessionHandle       → RaftInferSessionHandle
BrtExecutionPolicy     → RaftInferExecutionPolicy
BrtExecutionDiagnostics→ RaftInferExecutionDiagnostics
brt_*                  → raftinfer_*
```

Apply the same names in `cpp/src/c_api.cpp`, tests, and tools. Preserve numeric
enum values, struct field order, `struct_size` checks, ownership rules, and
error semantics.

- [ ] **Step 5: Update source-contract tests**

Replace old target, macro, include, type, and function expectations in:

```text
cpp/tests/c_api_source_test.cmake
cpp/tests/device_context_source_test.cmake
cpp/tests/qwen35_executor_source_test.cmake
```

Add negative assertions rejecting active old include and namespace spellings.

- [ ] **Step 6: Build and run the host-native test suite**

Run:

```bash
cmake -S . -B build/host -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=OFF \
  -DRAFTINFER_BUILD_TESTS=ON
cmake --build build/host
ctest --test-dir build/host --output-on-failure
```

Expected: all 18 host CTest tests pass and the library is named
`libraftinfer_cpp`.

- [ ] **Step 7: Commit the native rename**

```bash
git add CMakeLists.txt cmake cpp
git commit -m "refactor: rename native runtime to RAFTInfer"
```

---

### Task 3: Rename the Rust workspace, packages, FFI, and CLI

**Files:**
- Move: `rust/brt-sys/` → `rust/raftinfer-sys/`
- Move: `rust/brt-runtime/` → `rust/raftinfer-runtime/`
- Move: `rust/brt-cli/` → `rust/raftinfer-cli/`
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify: all Rust package manifests, sources, build scripts, and tests
- Test: all Rust workspace targets

**Interfaces:**
- Consumes: the renamed C ABI and `libraftinfer_cpp`.
- Produces: `raftinfer_sys`, `raftinfer_runtime`, `raftinfer-cli`, and the
  `raftinfer` executable.

- [ ] **Step 1: Move Rust package directories**

Run:

```bash
git mv rust/brt-sys rust/raftinfer-sys
git mv rust/brt-runtime rust/raftinfer-runtime
git mv rust/brt-cli rust/raftinfer-cli
```

- [ ] **Step 2: Rename workspace packages and dependencies**

Set:

```toml
[workspace]
resolver = "2"
members = [
  "rust/raftinfer-sys",
  "rust/raftinfer-runtime",
  "rust/raftinfer-cli",
]
```

Package names become `raftinfer-sys`, `raftinfer-runtime`, and
`raftinfer-cli`. The CLI manifest adds:

```toml
[[bin]]
name = "raftinfer"
path = "src/main.rs"
```

The sys manifest sets:

```toml
links = "raftinfer_cpp"
```

- [ ] **Step 3: Rename Rust imports and raw FFI declarations**

Apply:

```text
brt_sys          → raftinfer_sys
brt_runtime      → raftinfer_runtime
Brt*             → RaftInfer*
brt_*            → raftinfer_*
BRT_*            → RAFTINFER_*
brt_cpp          → raftinfer_cpp
```

Update `rust/raftinfer-sys/build.rs` to pass `RAFTINFER_ENABLE_CUDA`,
`RAFTINFER_BUILD_TESTS`, and `RAFTINFER_NATIVE_LIBRARY_TYPE` to CMake and link
`raftinfer_cpp`.

- [ ] **Step 4: Update CLI user-facing text and examples**

Error messages name RAFTInfer. Host-only errors read:

```text
generation requires a CUDA-enabled RAFTInfer build
benchmark requires a CUDA-enabled RAFTInfer build
```

The executable accepted by tests is `raftinfer`, not `brt-cli`.

- [ ] **Step 5: Format and test the Rust workspace**

Run:

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: all CLI, runtime, tokenizer, sys, and documentation tests pass.

- [ ] **Step 6: Commit the Rust rename**

```bash
git add Cargo.toml Cargo.lock rust
git commit -m "refactor: rename Rust workspace to RAFTInfer"
```

---

### Task 4: Rename scripts, containers, fixtures, and benchmark schema

**Files:**
- Modify: `containers/Dockerfile.dev`
- Modify: every tracked `scripts/*.sh`
- Modify: every tracked `tests/*.sh`
- Modify: JSON fixtures and golden files under `tests/`
- Modify: benchmark record C++/Rust tests
- Test: all repository shell tests

**Interfaces:**
- Consumes: renamed CLI, environment variables, CMake options, and native
  library.
- Produces: RAFTInfer-only automation and benchmark schema version 2.

- [ ] **Step 1: Rename automation contracts**

Apply to scripts, containers, and shell tests:

```text
BRT_*             → RAFTINFER_*
brt-cli           → raftinfer
brt-dev           → raftinfer-dev
brt-workspace     → raftinfer-workspace
brt-builds        → raftinfer-builds
brt-artifacts     → raftinfer-artifacts
/tmp/brt-         → /tmp/raftinfer-
BRT               → RAFTInfer
```

Keep script filenames such as `qwen35-parity.sh` because they describe the
operation rather than the former brand.

- [ ] **Step 2: Upgrade benchmark schema to version 2**

Update production emitters, gate scripts, and fixtures:

```json
{
  "schema_version": 2,
  "raftinfer": {},
  "provenance": {
    "raftinfer_model_sha256": "..."
  }
}
```

Remove `.brt` and `brt_model_sha256`. Gate scripts require exactly schema 2 and
the new fields. Add negative fixtures proving schema 1 and the old fields are
rejected.

- [ ] **Step 3: Rename GPU locks and diagnostics**

The cooperative lock becomes:

```text
/tmp/raftinfer-qwen35-gpu.lock
```

All refusal, parity, benchmark, and smoke messages name RAFTInfer while keeping
the existing safe behavior around unrelated GPU processes.

- [ ] **Step 4: Run script and schema tests**

Run:

```bash
bash -n scripts/*.sh tests/*.sh
tests/native-library-type-test.sh
tests/gpu-preflight-test.sh
tests/parity-script-test.sh
tests/benchmark-script-test.sh
tests/bf16-gate-script-test.sh
tests/prepare-qwen35-gguf-test.sh
tests/dockerfile-dev-test.sh
```

Expected: all positive paths pass and every intentional negative fixture is
rejected by the renamed diagnostics.

- [ ] **Step 5: Commit the automation and schema rename**

```bash
git add containers scripts tests cpp rust
git commit -m "refactor: rename automation and benchmark schema"
```

---

### Task 5: Create the GitHub project surface and reorganize documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/CODEOWNERS`
- Create: `.github/pull_request_template.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/ISSUE_TEMPLATE/config.yml`
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SECURITY.md`
- Create: `CHANGELOG.md`
- Create: `CITATION.cff`
- Create: `docs/architecture.md`
- Move: `docs/verification/m0.md` → `docs/verification/m0.md`
- Move: `docs/verification/m1.md` → `docs/verification/m1.md`
- Move: `docs/verification/m2.md` → `docs/verification/m2.md`
- Move: `docs/verification/m2a.md` → `docs/verification/m2a.md`
- Move: `docs/verification/m4.md` → `docs/verification/m4.md`
- Modify: `NOTICE`
- Modify: all current tracked Markdown files and relative links

**Interfaces:**
- Consumes: the renamed commands and project contracts.
- Produces: a complete GitHub community surface, portable documentation, and
  host-only continuous integration.

- [ ] **Step 1: Move milestone verification documents**

Use `git mv` for the five verification files and update every repository link
to `docs/verification/<milestone>.md`.

- [ ] **Step 2: Add host-only GitHub Actions CI**

Create `.github/workflows/ci.yml` with:

- `actions/checkout@v4`;
- `dtolnay/rust-toolchain@stable` with `rustfmt` and `clippy`;
- Ubuntu installation of CMake, Ninja, jq, and Python 3;
- `scripts/local-check.sh`;
- `cargo clippy --workspace --all-targets -- -D warnings`.

The workflow must not claim CUDA or performance validation.

- [ ] **Step 3: Add GitHub ownership and templates**

Set `.github/CODEOWNERS` to:

```text
* @charlesxu91
```

Issue forms require RAFTInfer version/commit, GPU, driver, CUDA version, model,
precision, reproduction commands, expected result, actual result, and logs
with secrets removed. The PR template requires host tests plus explicit
operator correctness, exact parity, performance, and provenance evidence when
the change touches GPU execution.

- [ ] **Step 4: Add community and release documents**

Write:

- `CONTRIBUTING.md` with host setup, RTX 50 gate order, commit expectations,
  and code review requirements;
- `CODE_OF_CONDUCT.md` using Contributor Covenant 2.1;
- `SECURITY.md` directing private reports through GitHub private vulnerability
  reporting and forbidding public exploit details before remediation;
- `CHANGELOG.md` using Keep a Changelog structure with an `Unreleased` section
  that records the breaking RAFTInfer rename;
- `CITATION.cff` using CFF 1.2.0, title `RAFTInfer`, version `0.1.0`,
  repository URL, Apache-2.0 license, and author name `charlesxu91`.

- [ ] **Step 5: Write the public architecture overview**

`docs/architecture.md` explains:

```text
Rust CLI/runtime
    ↓ coarse C ABI
C++ execution plan and session
    ↓
RAFT resources + RMM ownership
    ↓
cuBLASLt + custom CUDA kernels
```

It documents why Rust never calls individual kernels, how graph-safe fixed
workspace works, and how imported optimized kernels are gated without
publishing a bw24 performance comparison.

- [ ] **Step 6: Make documentation portable**

Replace personal absolute paths and local target names with:

```text
<repo>
<build-root>
<artifact-root>
<validation-root>
/path/to/model.gguf
```

Preserve cryptographic hashes, GPU model, CUDA/RAFT/RMM versions, and upstream
commits.

- [ ] **Step 7: Run documentation and brand checks**

Run:

```bash
python3 tests/readme-links-test.py
scripts/check-project-brand.sh
```

At this point the brand test may still fail only because README and benchmark
assets are created in Task 6. Any source, build, ABI, package, or current-doc
legacy hit is a Task 5 failure.

- [ ] **Step 8: Commit the GitHub project surface**

```bash
git add .github CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md \
  CHANGELOG.md CITATION.cff NOTICE docs
git commit -m "docs: add RAFTInfer GitHub project surface"
```

---

### Task 6: Publish bilingual README files and reproducible benchmark visuals

**Files:**
- Create: `README.zh-CN.md`
- Replace: `README.md`
- Create: `benchmarks/README.md`
- Create: `benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl`
- Create: `docs/benchmarks.md`
- Create: `docs/assets/qwen35-bf16-rtx5090.svg`
- Create: `tools/render_benchmark_chart.py`
- Create: `tests/benchmark-chart-test.py`
- Create: `tests/benchmark-asset-test.sh`
- Modify: `scripts/local-check.sh`

**Interfaces:**
- Consumes: schema-v2 formal benchmark JSONL and the renamed CLI commands.
- Produces: synchronized English/Chinese project entry points and a
  deterministic two-panel SVG.

- [ ] **Step 1: Commit the accepted formal result as schema version 2**

Transform the accepted M4 report without changing measured values:

```text
schema_version: 1 → 2
brt → raftinfer
brt_model_sha256 → raftinfer_model_sha256
```

Write the three JSONL records to
`benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl`. Preserve all provenance,
parity, execution, GPU, software, latency, throughput, variation, memory, and
ratio fields.

- [ ] **Step 2: Add the failing renderer behavior test**

Create `tests/benchmark-chart-test.py` with a hand-written three-record
schema-v2 fixture. Run the renderer as a subprocess and assert:

- it exits successfully for the valid fixture;
- the SVG is exactly 1200 by 620;
- both panel labels exist;
- the five RAFTInfer values, five llama.cpp values, and five ratios match the
  hand-derived fixture values;
- invalid schema, missing arm, failed parity, and failed performance-floor
  fixtures exit nonzero with actionable diagnostics.

Run:

```bash
python3 tests/benchmark-chart-test.py
```

Expected: FAIL because `tools/render_benchmark_chart.py` does not exist.

- [ ] **Step 3: Implement deterministic SVG rendering**

`tools/render_benchmark_chart.py`:

- accepts input JSONL and output SVG paths;
- requires exactly `pp128`, `pp512`, and `tg128_pp512`;
- requires schema 2, parity pass, and performance-floor pass;
- draws a 1200-by-620 SVG with Prefill and Generation panels;
- uses `#76B900` for RAFTInfer and `#64748B` for llama.cpp;
- prints tok/s above each bar and the RAFTInfer ratio above each group;
- writes no timestamps or machine-local paths.

- [ ] **Step 4: Run the renderer behavior test to green**

Run:

```bash
python3 tests/benchmark-chart-test.py
```

Expected: all valid and invalid behavior cases pass.

- [ ] **Step 5: Add the chart reproducibility test**

`tests/benchmark-asset-test.sh` renders to a temporary file and uses `cmp` to
compare it with `docs/assets/qwen35-bf16-rtx5090.svg`. It also asserts:

```text
6491.86
8391.24
87.46
85.01
84.98
1.967x
1.284x
1.038x
1.010x
1.009x
```

- [ ] **Step 6: Write the English README**

Use the approved section order. The top contains:

```markdown
# RAFTInfer

[简体中文](../../../README.zh-CN.md)

Correctness-gated RAFT/RMM inference with custom CUDA kernels for RTX
50-series GPUs.
```

Include the chart, exact benchmark protocol, scope, architecture, quick start,
renamed CLI commands, parity/performance reproduction, limitations, roadmap,
contribution, security, citation, and Apache-2.0 license.

- [ ] **Step 7: Write the Chinese README**

Mirror the English structure and facts. Link back with:

```markdown
[English](../../../README.md)
```

Do not translate command names, paths, model identifiers, revisions, or
measured values.

- [ ] **Step 8: Write benchmark documentation**

`benchmarks/README.md` defines the checked-in result policy and renderer.
`docs/benchmarks.md` records methodology, fixed artifact/revision, exact parity,
measurement counts, ratio definitions, memory meaning, shared-GPU preflight,
and limitations.

- [ ] **Step 9: Register and run asset tests**

Add `tests/benchmark-chart-test.py` and `tests/benchmark-asset-test.sh` to
`scripts/local-check.sh`, then run:

```bash
python3 tests/benchmark-chart-test.py
python3 tools/render_benchmark_chart.py \
  benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl \
  docs/assets/qwen35-bf16-rtx5090.svg
tests/benchmark-asset-test.sh
python3 tests/readme-links-test.py
tests/public-surface-test.sh
scripts/check-project-brand.sh
```

Expected: all pass.

- [ ] **Step 10: Commit README and benchmark assets**

```bash
git add README.md README.zh-CN.md benchmarks docs/assets \
  docs/benchmarks.md tools/render_benchmark_chart.py \
  tests/benchmark-chart-test.py tests/benchmark-asset-test.sh \
  scripts/local-check.sh
git commit -m "docs: publish RAFTInfer README and benchmarks"
```

---

### Task 7: Run the complete local quality gate and review the breaking rename

**Files:**
- Modify only files required by failures found in the complete gate
- Test: all host, Rust, script, documentation, and brand checks

**Interfaces:**
- Consumes: Tasks 1 through 6.
- Produces: one clean, review-approved candidate commit for target-GPU
  validation.

- [ ] **Step 1: Run the complete local gate**

Run:

```bash
scripts/local-check.sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
git diff --check
```

Expected: exit 0 throughout.

- [ ] **Step 2: Prove active old identifiers are absent**

Run:

```bash
tests/public-surface-test.sh
scripts/check-project-brand.sh
git grep -I -nE \
  'Blackwell RAFT Runtime|namespace brt|include/brt|Brt[A-Z]|brt_[A-Za-z]|BRT_[A-Z]|brt-(sys|runtime|cli|dev)|"brt_cpp"' \
  -- . \
  ':!docs/superpowers/specs/2026-07-30-raftinfer-open-source-renaming-design.md' \
  ':!docs/superpowers/plans/2026-07-30-raftinfer-open-source-renaming.md' \
  ':!CHANGELOG.md'
```

Expected: the test passes and the explicit grep prints nothing.

- [ ] **Step 3: Review API and schema consistency**

Review:

- every C declaration has the same renamed Rust FFI declaration;
- every CMake target used by Rust exists;
- every environment variable used by scripts is documented;
- schema-v2 emitters, gates, fixtures, result data, chart renderer, and README
  use identical fields;
- old binary/library names are not installed.

- [ ] **Step 4: Run a dedicated code review**

Request review focused on:

- incomplete mechanical rename;
- ABI type/function mismatches;
- build/link regressions;
- JSON schema drift;
- README claims not supported by evidence;
- GitHub workflow security and unpinned mutable inputs;
- accidental tracking of local artifacts or credentials.

Resolve all findings and rerun Step 1.

- [ ] **Step 5: Commit review fixes**

```bash
git add -A
git commit -m "fix: complete RAFTInfer rename review"
```

Skip this commit only when review produces no file changes.

---

### Task 8: Validate the renamed project on the RTX 5090 and refresh evidence

**Files:**
- Modify when formal measurements change: `benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl`
- Regenerate when data changes: `docs/assets/qwen35-bf16-rtx5090.svg`
- Modify when values change: `README.md`, `README.zh-CN.md`, `docs/benchmarks.md`
- Create target-local untracked build and evidence directories only

**Interfaces:**
- Consumes: the clean reviewed candidate and target GPU `192.168.124.8`.
- Produces: fresh CUDA tests, exact parity, performance gate, and final
  publication evidence.

- [ ] **Step 1: Archive the exact candidate commit**

Require a clean worktree, record `git rev-parse HEAD`, create a tracked-source
archive from that commit, and verify its SHA-256 locally and after transfer.
Never package `.git`, `.omx`, `.superpowers`, `.worktrees`, `build`, or
`target`.

Define the remote paths explicitly from caller-provided roots:

```bash
: "${RAFTINFER_VALIDATION_ROOT:?set RAFTINFER_VALIDATION_ROOT}"
: "${RAFTINFER_MODEL:?set RAFTINFER_MODEL}"
: "${RAFTINFER_RUNTIME_LIB:?set RAFTINFER_RUNTIME_LIB}"
candidate_sha="$(git rev-parse HEAD)"
candidate_short="$(printf '%s' "${candidate_sha}" | cut -c1-12)"
source_root="${RAFTINFER_VALIDATION_ROOT}/source-${candidate_short}"
build_root="${RAFTINFER_VALIDATION_ROOT}/build-${candidate_short}"
```

- [ ] **Step 2: Preflight the shared RTX 5090**

Read `nvidia-smi` compute applications, utilization, and memory. Do not stop
unrelated processes. Run only when `scripts/gpu-preflight.sh` accepts the GPU,
and use `/tmp/raftinfer-qwen35-gpu.lock`.

- [ ] **Step 3: Configure and build a clean Release artifact**

On the target, configure:

```bash
cmake -S "${source_root}" -B "${build_root}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAFTINFER_ENABLE_CUDA=ON \
  -DRAFTINFER_BUILD_TESTS=ON \
  -DRAFTINFER_NATIVE_LIBRARY_TYPE=SHARED
cmake --build "${build_root}" --parallel
```

Verify `libraftinfer_cpp.so` and the `raftinfer` CLI load the same formal native
library.

- [ ] **Step 4: Execute every CUDA/C++ test without opt-in skips**

Run under the cooperative GPU lock:

```bash
RAFTINFER_RUN_GPU_TESTS=1 \
LD_LIBRARY_PATH="${build_root}/cpp:${RAFTINFER_RUNTIME_LIB}" \
ctest --test-dir "${build_root}" --output-on-failure
```

Expected: all 25 tests execute and pass; zero required tests skip.

- [ ] **Step 5: Run exact greedy parity**

Run the renamed parity script with the formal Qwen3.5-9B BF16 artifact and
pinned llama.cpp server.

Expected:

- 4 records;
- 32 generated token IDs per record;
- 128 of 128 exact matches;
- online tiled attention;
- BF16 head-major KV cache;
- decode graph captured and replayed.

- [ ] **Step 6: Run the formal performance evaluator**

Run renamed PP128, PP512, and TG128 benchmarking with 5 warmups and 20 measured
iterations.

Expected:

- every RAFTInfer prefill and generation ratio is at least 1.0;
- at least one gated ratio is at least 1.1;
- RAFTInfer and llama.cpp coefficient of variation is at most 0.03;
- provenance, GPU, software, execution, and exact parity fields are complete;
- `scripts/qwen35-bf16-gate.sh` exits 0.

- [ ] **Step 7: Refresh committed evidence when measurements differ**

Convert the fresh formal report to the portable checked-in schema-v2 file. If
any displayed number changes after two stable same-window runs:

1. replace `benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl`;
2. regenerate `docs/assets/qwen35-bf16-rtx5090.svg`;
3. update both README files and `docs/benchmarks.md`;
4. rerun benchmark asset, link, brand, local, and performance gates;
5. commit with:

```bash
git add benchmarks docs/assets README.md README.zh-CN.md docs/benchmarks.md
git commit -m "docs: refresh RAFTInfer benchmark evidence"
```

- [ ] **Step 8: Record the target verification result**

Create `docs/verification/raftinfer-release.md` containing the verified commit,
archive hash, native library hash, CLI hash, 25-of-25 test result, parity hash,
benchmark hash, exact ratios, GPU, driver, CUDA, RAFT, RMM, model, llama.cpp
revision, and reproduction commands. Use portable target placeholders rather
than personal absolute paths.

Commit:

```bash
git add docs/verification/raftinfer-release.md
git commit -m "docs: verify RAFTInfer release candidate"
```

---

### Task 9: Merge the verified branch and publish `main` to GitHub

**Files:**
- Modify: local Git configuration for the `origin` remote
- No source changes unless final audit finds a documented issue

**Interfaces:**
- Consumes: clean, locally reviewed, GPU-verified candidate branch.
- Produces: identical local and GitHub `main` commits at
  `charlesxu91/RAFTInfer`.

- [ ] **Step 1: Run the pre-publication tracked-file audit**

Run:

```bash
git status --short
git ls-files | grep -E '(^|/)(build|target|\\.omx|\\.superpowers|\\.worktrees)(/|$)' && exit 1 || true
git grep -I -nE '/Users/|/home/charles|192\\.168\\.' -- . && exit 1 || true
git grep -I -nE '(ghp_|github_pat_|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY)' -- . && exit 1 || true
scripts/local-check.sh
```

Expected: clean worktree, no local artifacts, no personal machine paths, no
credential patterns, and all local tests pass.

- [ ] **Step 2: Resolve the destination remote safely**

Inspect:

```bash
git remote -v
git ls-remote https://github.com/charlesxu91/RAFTInfer.git
```

If `origin` is absent:

```bash
git remote add origin https://github.com/charlesxu91/RAFTInfer.git
```

If `origin` exists with another URL:

```bash
git remote set-url origin https://github.com/charlesxu91/RAFTInfer.git
```

If remote refs exist, fetch them and compare history before merging. Do not use
`--force`.

- [ ] **Step 3: Fast-forward local `main` to the verified branch**

From the primary worktree:

```bash
git switch main
git merge --ff-only codex/raftinfer-open-source
```

Rerun `git status --short` and verify the expected final commit.

- [ ] **Step 4: Push `main`**

Run:

```bash
git push -u origin main
```

Do not push temporary worktree branches.

- [ ] **Step 5: Verify remote identity**

Run:

```bash
local_head="$(git rev-parse main)"
remote_head="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
test "${local_head}" = "${remote_head}"
```

Expected: exact SHA equality.

- [ ] **Step 6: Final completion audit**

Confirm:

- local and remote `main` match;
- the GitHub landing page has English README and a working Chinese link;
- the benchmark SVG and badges resolve;
- CI is present and scoped to host validation;
- all acceptance criteria in the design are satisfied;
- no required work remains.
