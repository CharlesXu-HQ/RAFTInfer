## Task 8 Report: Release Assert Test Guard

Status: DONE

### Scope

- Addressed the M1 final review minor: C++ tests relied on `assert()`, so a
  Release build with `NDEBUG` could false-pass.
- Changed only C++ test build wiring and C++ test sources.
- Did not modify production C++ or CUDA sources; prior RTX 5090 validation
  evidence remains applicable.

### RED Evidence

- Added `cpp/tests/assert_enabled.hpp` to one assert-based test first.
- `cmake -S . -B build/release-test -G Ninja -DBRT_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`: PASS.
- `cmake --build build/release-test`: expected FAIL.
- Failure showed `clang++ ... -O3 -DNDEBUG ... c_api_test.cpp` followed by
  `#error "C++ tests rely on assert(); test targets must compile with NDEBUG undefined."`

### Fix

- Added `brt_add_assert_test()` helper in `cpp/CMakeLists.txt`.
- The helper applies only to C++ test executables and undefines `NDEBUG`:
  - GNU/Clang: `-UNDEBUG`
  - MSVC: `/UNDEBUG`
- Converted all assert-based compiled C++ tests to use the helper.
- Included `assert_enabled.hpp` in all assert-based compiled C++ tests:
  - `benchmark_record_test.cpp`
  - `c_api_test.cpp`
  - `correctness_test.cpp`
  - `operator_registry_test.cpp`
  - `reference_operators_test.cpp`
  - `tensor_validation_test.cpp`
  - `workspace_layout_test.cpp`
- Source-only CMake tests were left unchanged.

### Verification

- `cmake --build build/release-test`: PASS.
- `ctest --test-dir build/release-test --output-on-failure`: PASS, 9/9 tests.
- `rg -n "brt_c_api_test.dir/tests/c_api_test.cpp.o|UNDEBUG|DNDEBUG" build/release-test/build.ninja`:
  confirmed test compile flags include `-O3 -DNDEBUG -std=gnu++20 -UNDEBUG`,
  while non-test Release targets retain normal `-DNDEBUG` flags.
- `scripts/local-check.sh`: PASS.

### Concerns

- None remaining.
