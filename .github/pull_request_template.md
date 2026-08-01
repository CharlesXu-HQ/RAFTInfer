## Summary

Describe the change and its user-visible effect.

## Host validation

- [ ] `scripts/local-check.sh` passed.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passed.
- [ ] Documentation and migration notes are updated where applicable.

## GPU execution changes

If this change touches GPU execution, provide all applicable evidence below.

- [ ] Operator correctness evidence against the independent reference is attached.
- [ ] Exact greedy-token parity evidence is attached.
- [ ] RTX 50 performance-gate evidence is attached; no hosted CI result is represented as CUDA or performance validation.
- [ ] Kernel license, upstream provenance, local modifications, and reuse eligibility are documented.
