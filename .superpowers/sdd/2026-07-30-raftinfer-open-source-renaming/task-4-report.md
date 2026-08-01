# Task 4 report: automation and benchmark schema rename

## RED

Updated the benchmark and BF16 gate fixture expectations before production
automation changes: schema v2, `.raftinfer`, and
`provenance.raftinfer_model_sha256`. `tests/benchmark-script-test.sh` then
failed with exit code 43 and `provenance JSON is not a pinned BF16 artifact`,
because the prior gate accepted schema v1.

Updated the C API tokenizer-wire assertion to `RIFTOK`; the focused C++ test
failed because the native serializer was still producing `BRTTOK`. Updated the
Qwen fixture parser expectation to `RIFQ35F1`; its focused test then failed
against the old binary fixture magic.

## GREEN

- Renamed scripts, shell tests, containers-facing tags/paths, environments,
  locks, diagnostics, and fixture references to RAFTInfer-only spellings.
- Benchmark/parity/provenance records now require and emit schema version 2,
  `.raftinfer`, and `provenance.raftinfer_model_sha256`.
- Added executable negative fixtures: schema-v1 provenance and the old
  `brt_model_sha256` field are rejected.
- Renamed tokenizer wire magic to `RIFTOK\\0` and Qwen fixture magic to
  `RIFQ35F1`; updated the native producer/parser, Rust parser/tests, exporter,
  binary fixture, and fixture checksum.
- Kept GPU preflight's refusal of unrelated active compute processes; only
  contract naming changed.

## Commands and results

```text
bash -n scripts/*.sh tests/*.sh                                      PASS
tests/native-library-type-test.sh                                    PASS
tests/gpu-preflight-test.sh                                          PASS
tests/parity-script-test.sh                                          PASS
tests/benchmark-script-test.sh                                       PASS
tests/bf16-gate-script-test.sh                                       PASS
tests/prepare-qwen35-gguf-test.sh                                    PASS
tests/dockerfile-dev-test.sh                                         PASS
for test_script in tests/*.sh; do "$test_script"; done              PASS
cmake ... -DRAFTINFER_ENABLE_CUDA=OFF; ctest focused C++ tests       PASS
cargo test -p raftinfer-runtime tokenizer                             PASS
cargo test -p raftinfer-cli benchmark                                PASS
```

## Concerns

The Qwen fixture was intentionally changed in place (same length/header
layout); its SHA-256 is consequently updated. The retained old-brand strings
are limited to the deliberate negative fixture and legacy-detection tests,
which make rejected inputs and scans observable; production automation,
schemas, and wire producers/parsers have no legacy spelling.
