# Task 3 evidence report

## Scope delivered

- Added a deterministic metadata-only operator registry under `cpp/registry`.
- Added comparable tensor/operator signature value types and complete `std::hash` coverage for all signature fields.
- Added kernel capability matching for operator kind, execution regime, architecture range, dtype, quantization, rank, alignment, shape bounds, graph capture safety, determinism, and workspace bytes.
- Added deterministic dispatch selection: priority descending, then registration name ascending.
- Added stable resolve caching that returns the same registration object for repeated signatures.
- Added duplicate registration-name rejection and `DispatchError` with per-candidate rejection reasons.
- Registered `brt_operator_registry_test` in host CMake.

## TDD evidence

1. RED: Added `cpp/tests/operator_registry_test.cpp` and the CMake test target before creating `cpp/registry/operator_registry.hpp`.
2. RED observed: `cmake --build build/host --target brt_operator_registry_test` failed at compile time with `fatal error: '../registry/operator_registry.hpp' file not found`.
3. GREEN: Added `operator_registry.hpp/.cpp`, registered the source in `brt_cpp`, and iterated until the target built and the focused registry test passed.
4. Test correction during GREEN: Split the nondeterministic rejection assertion into a separate registry because the deterministic fallback correctly satisfies deterministic requests when enough workspace is available.

## Final verification

All commands were run from `.worktrees/m1-registry-correctness`:

```text
cmake --build build/host --target brt_operator_registry_test
  PASS
ctest --test-dir build/host -R brt_operator_registry_test --repeat until-fail:2 --output-on-failure
  PASS (2/2)
cmake --build build/host
  PASS
ctest --test-dir build/host --output-on-failure
  PASS (6/6)
git diff --check
  PASS
```

## Commit

- `feat: add deterministic operator registry`

## Remaining concerns

- No Rust per-op FFI was added, and no `bw24` import was introduced.

## Review fix: per-input capability constraints

- Fixed the review defect where `KernelCapability::rejection_reason` validated only `signature.inputs.front()`.
- Replaced scalar tensor capability fields with `std::vector<TensorConstraint>` aligned to `OperatorSignature::inputs`.
- Added input-count mismatch rejection before per-input validation.
- Added per-input dtype, quantization, rank, alignment, and shape-bound validation, with rejection reasons labeled by input index.

### Review-fix TDD evidence

1. RED: Added two-input registry tests using `TensorConstraint` before production support existed.
2. RED observed: `cmake --build build/host --target brt_operator_registry_test` failed with `no type named 'TensorConstraint' in namespace 'brt'`.
3. GREEN: Added `TensorConstraint`, changed `KernelCapability` to hold per-input constraints, and updated matching to validate every input.
4. Regression coverage verifies correct selection when only input 2 differs, rejection reasons for input 2 dtype mismatch, and input-count mismatch rejection.

### Review-fix verification

```text
cmake --build build/host --target brt_operator_registry_test
  PASS
ctest --test-dir build/host -R brt_operator_registry_test --output-on-failure
  PASS (1/1)
ctest --test-dir build/host -R brt_operator_registry_test --repeat until-fail:2 --output-on-failure
  PASS (2/2)
cmake --build build/host
  PASS
ctest --test-dir build/host --output-on-failure
  PASS (6/6)
```
