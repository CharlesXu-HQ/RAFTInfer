# RAFTInfer Open-Source Renaming and Repository Design

Date: 2026-07-30

## 1. Objective

Rename the project from Blackwell RAFT Runtime and its `brt` technical
identifiers to **RAFTInfer**, reorganize the repository as a publication-ready
C++/CUDA and Rust open-source project, publish synchronized English and Chinese
README files, present reproducible performance evidence against llama.cpp, and
push the verified `main` branch to:

```text
https://github.com/charlesxu91/RAFTInfer.git
```

The rename is intentionally breaking. The repository will not retain aliases,
fallback environment variables, wrapper symbols, compatibility headers, or
deprecated package names for the former project identity.

## 2. Product Positioning

RAFTInfer is a correctness-gated Qwen3.5 inference runtime for NVIDIA RTX
50-series Blackwell GPUs. RAFT and RMM own the common GPU resource, stream, and
memory foundation. Performance-critical operations remain project-owned
C++/CUDA implementations or proven external kernels that pass the project's
license, provenance, algorithmic, numerical, exact-token, and RTX 50 performance
gates.

The public value proposition has four parts:

1. **A reusable foundation:** RAFT/RMM resource ownership is separated from
   model-specific CUDA scheduling.
2. **Measured performance:** every published comparison uses pinned artifacts,
   identical prompts, same-machine execution, warmups, repeated measurements,
   and a machine-readable gate.
3. **Correctness before speed:** operator reference tests and exact greedy token
   parity are prerequisites for performance publication.
4. **A deliberate C++/Rust boundary:** C++ owns GPU state and coarse execution;
   Rust owns the safe runtime, tokenizer, CLI, and orchestration without
   per-operator FFI.

The public scope remains RTX 50-series GPUs only. RTX 40-series support is not
part of this release.

## 3. Repository Structure

The existing language-oriented top-level structure is retained because it is
clear for a mixed C++/CUDA and Rust repository. Core implementation directories
will not be moved merely for cosmetic conformity.

```text
RAFTInfer/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── feature_request.yml
│   │   └── config.yml
│   ├── workflows/
│   │   └── ci.yml
│   ├── CODEOWNERS
│   └── pull_request_template.md
├── benchmarks/
│   ├── README.md
│   └── results/
│       └── qwen35-9b-bf16-rtx5090.jsonl
├── cmake/
├── containers/
├── cpp/
│   ├── benchmarks/
│   ├── execution/
│   ├── foundation/
│   ├── include/raftinfer/
│   ├── kernels/
│   ├── model/
│   ├── operators/
│   ├── reference/
│   ├── registry/
│   ├── src/
│   ├── tests/
│   └── tools/
├── docs/
│   ├── assets/
│   │   └── qwen35-bf16-rtx5090.svg
│   ├── architecture.md
│   ├── benchmarks.md
│   ├── provenance/
│   ├── superpowers/
│   └── verification/
├── rust/
│   ├── raftinfer-cli/
│   ├── raftinfer-runtime/
│   └── raftinfer-sys/
├── scripts/
├── tests/
├── tools/
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE
├── README.md
├── README.zh-CN.md
└── SECURITY.md
```

Milestone verification documents move from the root of `docs/` to
`docs/verification/`. All relative links are updated. Machine-specific absolute
paths are replaced with portable placeholders or artifact descriptions; hashes
and pinned upstream revisions remain intact.

## 4. Breaking Rename Contract

The following transformations are complete, repository-wide contracts:

| Surface | Old | New |
| --- | --- | --- |
| C++ namespace | `brt` | `raftinfer` |
| Public include root | `cpp/include/brt` | `cpp/include/raftinfer` |
| C ABI types | `Brt*` | `RaftInfer*` |
| C ABI functions | `brt_*` | `raftinfer_*` |
| CMake project | `brt` | `raftinfer` |
| Native target/library | `brt_cpp` | `raftinfer_cpp` |
| CMake options | `BRT_*` | `RAFTINFER_*` |
| Rust packages | `brt-*` | `raftinfer-*` |
| Rust imports | `brt_*` | `raftinfer_*` |
| CLI binary | `brt-cli` | `raftinfer` |
| Runtime environment | `BRT_*` | `RAFTINFER_*` |
| Docker image | `brt-dev` | `raftinfer-dev` |
| Lock and temporary names | `brt-*` | `raftinfer-*` |
| Benchmark engine key | `.brt` | `.raftinfer` |
| Model checksum key | `brt_model_sha256` | `raftinfer_model_sha256` |

The benchmark JSON schema increments from version 1 to version 2 because the
engine object and checksum field change. Tests reject the old schema and old
field names. No v1 compatibility reader is added.

Tracked source, tests, scripts, current documentation, CI, examples, fixtures,
and generated public assets must contain no old project identifier after the
rename. The only prose exceptions are this migration design, its implementation
plan, and the changelog, where the former identifiers may appear solely to
describe historical old-to-new mappings. Git history is not rewritten.

References to `bw24` remain only where they describe provenance or the external
kernel reuse policy. The README does not compare RAFTInfer performance with
`bw24`.

## 5. README Design

`README.md` is the default English landing page. `README.zh-CN.md` is its
Chinese counterpart. Each begins with a language switch and has the same
section order:

1. project title, concise positioning, and badges;
2. current scope and release status;
3. why RAFTInfer exists;
4. formal performance chart and key numbers;
5. correctness contract;
6. architecture overview;
7. supported hardware, model, and precision;
8. quick start and build commands;
9. CLI example;
10. parity and benchmark reproduction;
11. repository layout;
12. roadmap and known limitations;
13. contributing, security, citation, and license.

README claims distinguish measured facts from planned work. M4 is the current
accepted BF16 path. Unsupported model, precision, server, batching, quantized,
or GPU features are not advertised as available.

## 6. Performance Evidence and Chart

The README chart compares only RAFTInfer and the pinned llama.cpp reference on
the same desktop RTX 5090, using the same Qwen3.5-9B BF16 GGUF artifact and
identical benchmark arms.

The checked-in benchmark result contains:

| Arm | RAFTInfer | llama.cpp | Ratio |
| --- | ---: | ---: | ---: |
| PP128 prefill | 6491.86 tok/s | 3300.20 tok/s | 1.9671x |
| PP512 prefill | 8391.24 tok/s | 6534.15 tok/s | 1.2842x |
| PP128 generation | 87.46 tok/s | 84.29 tok/s | 1.0376x |
| PP512 generation | 85.01 tok/s | 84.19 tok/s | 1.0097x |
| TG128 at PP512 | 84.98 tok/s | 84.18 tok/s | 1.0095x |

The SVG has two panels:

- Prefill: grouped bars for PP128 and PP512.
- Generation: grouped bars for PP128, PP512, and TG128 at PP512.

Every bar displays tok/s. Each RAFTInfer group displays its ratio to llama.cpp.
The caption records:

- NVIDIA GeForce RTX 5090;
- CUDA 13.2;
- RAFT 26.06 and RMM 26.06;
- Qwen3.5-9B BF16;
- 5 warmup iterations and 20 measured iterations;
- exact parity of 4 records by 32 generated tokens;
- pinned llama.cpp revision
  `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`.

The SVG is rendered deterministically from
`benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl`. A repository test validates
the source hash and displayed values so the chart cannot silently diverge from
the evidence.

## 7. GitHub Project Files

The repository adds:

- host-only CI for CMake, CTest, Rust format, Rust clippy, Rust tests, shell
  syntax, repository-brand checks, README link checks, and benchmark asset
  validation;
- issue forms for reproducible bugs and bounded feature requests;
- a pull request checklist covering tests, correctness evidence, performance
  evidence, provenance, and documentation;
- `CODEOWNERS` assigning the repository to `@charlesxu91`;
- contribution instructions that distinguish host CI from required RTX 50 GPU
  validation;
- a security policy that uses GitHub private vulnerability reporting rather
  than publishing personal contact details;
- the Contributor Covenant code of conduct;
- a Keep a Changelog-compatible changelog beginning with the current
  unreleased RAFTInfer rename;
- a `CITATION.cff` record naming RAFTInfer, the Apache-2.0 license, and the
  GitHub repository.

GitHub Actions does not claim CUDA correctness on hosted runners. CUDA, exact
parity, and performance gates remain documented target-GPU commands.

## 8. Testing Strategy

The rename is behavior-preserving except for the deliberately breaking public
names and benchmark schema. Existing tests lock computational behavior before
mechanical renaming.

A repository-brand test is added first and initially fails on:

- old namespaces, types, functions, packages, environment variables, targets,
  paths, benchmark keys, and project prose;
- missing bilingual README files or language links;
- missing community files;
- old schema version or missing schema-v2 fields;
- tracked personal machine paths.

After implementation, validation proceeds in this order:

1. repository-brand and documentation link tests;
2. shell syntax and benchmark chart consistency tests;
3. host CMake build and all host CTest tests;
4. Rust formatting, clippy, and all workspace tests;
5. `scripts/local-check.sh`;
6. CUDA Release build for `sm_120a`;
7. all 25 CUDA/C++ tests with `RAFTINFER_RUN_GPU_TESTS=1`;
8. 4-by-32 exact greedy parity;
9. PP128, PP512, and TG128 formal performance gate;
10. clean-worktree and tracked-file audit.

Published performance must remain at least 1.0x llama.cpp for all gated
prefill and generation metrics, with exact parity and test success.

## 9. Publication Procedure

The verified local `main` branch is published to:

```text
https://github.com/charlesxu91/RAFTInfer.git
```

Before adding or updating `origin`, the workflow reads the remote references.
If the repository is empty, local `main` is pushed normally. If the repository
already contains commits, they are fetched and inspected before integration.
The workflow never force-pushes over unknown remote history.

The publication audit ensures that `.omx/`, `.superpowers/`, `.worktrees/`,
`build/`, `target/`, models, target-machine artifacts, credentials, and
machine-local evidence are not tracked. After push, the remote `main` commit is
resolved and compared with the verified local commit.

## 10. Acceptance Criteria

The work is complete only when:

1. all active public and internal project identifiers use RAFTInfer naming;
2. no compatibility layer for the old name exists;
3. the repository structure and GitHub community files match this design;
4. English and Chinese README files are synchronized and accurate;
5. the committed chart matches the committed formal benchmark evidence;
6. README performance comparisons include llama.cpp only;
7. host and RTX 5090 validation passes without skipped required GPU tests;
8. exact greedy parity remains 128 of 128 generated tokens;
9. PP128, PP512, and TG128 remain at or above the 1.0x performance floor;
10. the final clean `main` commit is present at
    `github.com/charlesxu91/RAFTInfer`.
