# Contributing to RAFTInfer

Thank you for improving RAFTInfer. Please keep changes small, reviewable, and
grounded in the project's correctness-first contract.

## Development and review

Run the host checks before opening a pull request:

```bash
scripts/local-check.sh
cargo clippy --workspace --all-targets -- -D warnings
```

Use focused, imperative commits and explain the affected contract in the pull
request. Review requires readable code, host-test evidence, and updated
documentation when behavior or public interfaces change.

## RTX 50 validation gate

Hosted CI is host-only. It does not validate CUDA correctness, exact parity, or
performance. Changes to GPU execution proceed in this order: independent
operator correctness, exact greedy-token parity, then the RTX 50 performance
gate. Attach the resulting target-GPU evidence to the pull request.

Imported optimized kernels may be considered only after their license,
provenance, algorithmic and numerical correctness, and RTX 50 performance have
all passed review. Record source path, upstream revision, license, imported
files, and local modifications. Do not publish comparisons against `bw24`.
