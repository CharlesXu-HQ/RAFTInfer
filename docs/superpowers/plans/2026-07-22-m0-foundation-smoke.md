# M0 Foundation and Full-Stack Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible C++20/CUDA plus Rust project spine in which Rust creates a C++ engine, the C++ engine owns RAFT 26.06 resources and an RMM 26.06 pool, and a target RTX 5090 executes and verifies one GPU smoke kernel.

**Architecture:** macOS builds a host-only C++ stub and Rust safe wrapper to validate the ABI without CUDA. GPU builds run in the pinned `rapidsai/base:26.06-cuda13-py3.13` container, where C++ owns RAFT/RMM/CUDA state and exposes opaque handles through a stable `raftinfer_*` C ABI. The remote smoke runner checks shared-GPU state before starting and never modifies system GPU settings or interferes with other processes.

**Tech Stack:** C++20, CUDA 13.x, RAFT 26.06, RMM 26.06, CMake 3.26.4+, Ninja, Rust 1.96.0, Cargo, Docker 29+, NVIDIA Container Toolkit 1.19+, RTX 50 `sm_120a`.

This plan intentionally covers M0 only. M1 through M5 each require a separate implementation plan after M0 produces verified build and runtime interfaces.

## Global Constraints

- v0.1 targets RTX 50-series consumer Blackwell only; compile CUDA code for `sm_120a`.
- RAFT/RMM own common GPU resource and memory infrastructure; custom CUDA owns hot kernels.
- C++ owns every GPU pointer, stream, event, allocation, and graph; Rust receives opaque handles only.
- Public C symbols use the `raftinfer_` prefix and public headers live under `cpp/include/raftinfer/`.
- C++ exceptions must not cross the C ABI.
- No GPU allocation occurs in the host-only build.
- The target host is `<validation-root>`, user `charles`; authentication secrets never enter repository files, scripts, logs, or evidence.
- Never stop or alter unrelated GPU processes. Abort a GPU test if the shared-GPU preflight detects compute activity or insufficient memory headroom.
- Use Apache-2.0 for original project code and retain third-party notices.
- M0 does not load a model, parse GGUF, implement operators, or benchmark inference.

## Planned file structure

```text
/
├── .gitignore                         Build, Cargo, IDE, and evidence exclusions
├── CMakeLists.txt                     Top-level C++/CUDA build and options
├── Cargo.toml                         Rust workspace
├── LICENSE                            Apache License 2.0
├── NOTICE                             Project and third-party notice entry point
├── cmake/
│   └── RaftInferOptions.cmake               CUDA/backend configuration checks
├── cpp/
│   ├── CMakeLists.txt                 C++ library, tests, and smoke executable
│   ├── include/raftinfer/
│   │   ├── c_api.h                    Stable opaque-handle C ABI
│   │   └── status.h                   ABI-safe status codes
│   ├── src/
│   │   ├── status.cpp                 Non-empty static-library translation unit
│   │   ├── c_api.cpp                  Exception barrier and engine lifecycle
│   │   ├── engine.cpp                 Host/GPU engine implementation
│   │   └── engine.hpp                 Private Engine class
│   ├── foundation/
│   │   ├── device_context.cu          RAFT resources and RMM pool ownership
│   │   └── device_context.hpp         Private DeviceContext interface
│   ├── kernels/
│   │   └── smoke.cu                   Minimal custom CUDA smoke kernel
│   ├── tests/
│   │   └── c_api_test.cpp             Host-only ABI lifecycle tests
│   └── tools/
│       └── raftinfer_smoke.cpp              GPU smoke CLI
├── rust/
│   ├── raftinfer-sys/
│   │   ├── Cargo.toml                 Raw ABI crate
│   │   ├── build.rs                   CMake build and native link configuration
│   │   └── src/lib.rs                 Raw FFI declarations
│   ├── raftinfer-runtime/
│   │   ├── Cargo.toml                 Safe Rust API
│   │   ├── src/lib.rs                 RAII Engine wrapper
│   │   └── tests/engine.rs            Rust lifecycle/error tests
│   └── raftinfer-cli/
│       ├── Cargo.toml                 Smoke CLI crate
│       └── src/main.rs                Host/GPU smoke command
├── containers/
│   └── Dockerfile.dev                 Pinned RAPIDS/CUDA/Rust build image
├── scripts/
│   ├── gpu-preflight.sh               Shared-GPU safety gate
│   ├── gpu-smoke.sh                   Containerized GPU build/run
│   ├── local-check.sh                 Host-only build and Rust tests
│   └── sync-target.sh                 Non-destructive target source sync
├── tests/golden/
│   └── smoke_result.json              Expected GPU smoke result schema/value
└── docs/provenance/
    └── dependencies.md                Exact M0 dependency sources and licenses
```

---

### Task 1: Host-only C++ build spine and status contract

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `NOTICE`
- Create: `CMakeLists.txt`
- Create: `cmake/RaftInferOptions.cmake`
- Create: `cpp/CMakeLists.txt`
- Create: `cpp/include/raftinfer/status.h`
- Create: `cpp/src/status.cpp`
- Create: `cpp/tests/c_api_test.cpp`

**Interfaces:**
- Produces: `RaftInferStatusCode`, `RaftInferStatus`, `RAFTINFER_ENABLE_CUDA`, and the `raftinfer_cpp` build target used by every later task.
- Consumes: no project code.

- [ ] **Step 1: Add repository metadata and build exclusions**

Create `.gitignore` with:

```gitignore
/build/
/target/
/.cache/
/.idea/
/.vscode/
*.swp
*.swo
compile_commands.json
/evidence/local/
```

Create `LICENSE` using the complete Apache License 2.0 text published at `https://www.apache.org/licenses/LICENSE-2.0.txt`. Create `NOTICE` with:

```text
RAFTInfer
Copyright 2026 RAFTINFER contributors

This product includes software developed by third parties. Their notices and
license texts are recorded under THIRD_PARTY_LICENSES/ and docs/provenance/.
```

- [ ] **Step 2: Write the failing status-contract test**

Create `cpp/tests/c_api_test.cpp`:

```cpp
#include <raftinfer/status.h>

#include <cassert>
#include <type_traits>

int main() {
  static_assert(std::is_standard_layout_v<RaftInferStatus>);
  static_assert(std::is_trivially_copyable_v<RaftInferStatus>);
  RaftInferStatus ok{RAFTINFER_STATUS_OK, nullptr};
  assert(ok.code == RAFTINFER_STATUS_OK);
  assert(ok.message == nullptr);
  return 0;
}
```

- [ ] **Step 3: Add the top-level CMake configuration and verify the test fails**

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.26.4)
project(raftinfer VERSION 0.1.0 LANGUAGES CXX)

include(cmake/RaftInferOptions.cmake)
add_subdirectory(cpp)
```

Create `cmake/RaftInferOptions.cmake`:

```cmake
option(RAFTINFER_ENABLE_CUDA "Build the RAFT/RMM CUDA backend" OFF)
option(RAFTINFER_BUILD_TESTS "Build RAFTINFER tests" ON)

if(RAFTINFER_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures" FORCE)
endif()
```

Create `cpp/CMakeLists.txt`:

```cmake
add_library(raftinfer_cpp STATIC src/status.cpp)
target_compile_features(raftinfer_cpp PUBLIC cxx_std_20)
target_include_directories(raftinfer_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(RAFTINFER_BUILD_TESTS)
  enable_testing()
  add_executable(raftinfer_c_api_test tests/c_api_test.cpp)
  target_link_libraries(raftinfer_c_api_test PRIVATE raftinfer_cpp)
  add_test(NAME raftinfer_c_api_test COMMAND raftinfer_c_api_test)
endif()
```

Create `cpp/src/status.cpp`:

```cpp
#include <raftinfer/status.h>

#include <type_traits>

static_assert(std::is_standard_layout_v<RaftInferStatus>);
static_assert(std::is_trivially_copyable_v<RaftInferStatus>);
```

Run:

```bash
cmake -S . -B build/host -G Ninja -DRAFTINFER_ENABLE_CUDA=OFF
cmake --build build/host
```

Expected: compile fails because `raftinfer/status.h` does not exist.

- [ ] **Step 4: Implement the ABI-safe status types**

Create `cpp/include/raftinfer/status.h`:

```cpp
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum RaftInferStatusCode {
  RAFTINFER_STATUS_OK = 0,
  RAFTINFER_STATUS_INVALID_ARGUMENT = 1,
  RAFTINFER_STATUS_UNAVAILABLE = 2,
  RAFTINFER_STATUS_CUDA_ERROR = 3,
  RAFTINFER_STATUS_INTERNAL = 4
} RaftInferStatusCode;

typedef struct RaftInferStatus {
  RaftInferStatusCode code;
  const char* message;
} RaftInferStatus;

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 5: Build and run the host test**

Run:

```bash
cmake --build build/host
ctest --test-dir build/host --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 6: Commit the build spine**

```bash
git add .gitignore LICENSE NOTICE CMakeLists.txt cmake/RaftInferOptions.cmake cpp/CMakeLists.txt cpp/include/raftinfer/status.h cpp/src/status.cpp cpp/tests/c_api_test.cpp
git commit -m "build: add host-only C++ project spine"
```

---

### Task 2: Opaque C ABI and host engine lifecycle

**Files:**
- Create: `cpp/include/raftinfer/c_api.h`
- Create: `cpp/src/engine.hpp`
- Create: `cpp/src/engine.cpp`
- Create: `cpp/src/c_api.cpp`
- Modify: `cpp/CMakeLists.txt`
- Modify: `cpp/tests/c_api_test.cpp`

**Interfaces:**
- Consumes: `RaftInferStatus` from Task 1.
- Produces: `RaftInferEngineHandle`, `RaftInferEngineConfig`, `raftinfer_engine_create`, `raftinfer_engine_destroy`, `raftinfer_engine_is_cuda_enabled`, and `raftinfer_last_error_message` for the Rust wrapper.

- [ ] **Step 1: Extend the test with lifecycle and argument failures**

Replace `cpp/tests/c_api_test.cpp` with:

```cpp
#include <raftinfer/c_api.h>

#include <cassert>
#include <cstring>

int main() {
  RaftInferEngineConfig config{};
  config.struct_size = sizeof(RaftInferEngineConfig);
  config.device_id = 0;
  config.initial_pool_bytes = 64U * 1024U * 1024U;

  assert(raftinfer_engine_create(nullptr, nullptr).code == RAFTINFER_STATUS_INVALID_ARGUMENT);

  RaftInferEngineHandle* engine = nullptr;
  RaftInferStatus status = raftinfer_engine_create(&config, &engine);
  assert(status.code == RAFTINFER_STATUS_OK);
  assert(engine != nullptr);
  assert(raftinfer_engine_is_cuda_enabled(engine) == 0);

  raftinfer_engine_destroy(engine);
  assert(std::strlen(raftinfer_last_error_message()) == 0);
  return 0;
}
```

- [ ] **Step 2: Run the test and verify the missing ABI fails compilation**

Run:

```bash
cmake --build build/host
```

Expected: compile fails because `raftinfer/c_api.h` does not exist.

- [ ] **Step 3: Define the public ABI**

Create `cpp/include/raftinfer/c_api.h`:

```cpp
#pragma once

#include <raftinfer/status.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RaftInferEngineHandle RaftInferEngineHandle;

typedef struct RaftInferEngineConfig {
  size_t struct_size;
  int32_t device_id;
  uint64_t initial_pool_bytes;
} RaftInferEngineConfig;

RaftInferStatus raftinfer_engine_create(const RaftInferEngineConfig* config, RaftInferEngineHandle** out_engine);
void raftinfer_engine_destroy(RaftInferEngineHandle* engine);
int32_t raftinfer_engine_is_cuda_enabled(const RaftInferEngineHandle* engine);
const char* raftinfer_last_error_message(void);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 4: Implement the private host engine and exception barrier**

Create `cpp/src/engine.hpp`:

```cpp
#pragma once

#include <raftinfer/c_api.h>

namespace raftinfer {

class Engine {
 public:
  explicit Engine(const RaftInferEngineConfig& config);
  bool cuda_enabled() const noexcept;

 private:
  int device_id_;
  uint64_t initial_pool_bytes_;
};

}  // namespace raftinfer
```

Create `cpp/src/engine.cpp`:

```cpp
#include "engine.hpp"

#include <stdexcept>

namespace raftinfer {

Engine::Engine(const RaftInferEngineConfig& config)
    : device_id_(config.device_id), initial_pool_bytes_(config.initial_pool_bytes) {
  if (device_id_ < 0) throw std::invalid_argument("device_id must be non-negative");
  if (initial_pool_bytes_ == 0) throw std::invalid_argument("initial_pool_bytes must be non-zero");
}

bool Engine::cuda_enabled() const noexcept {
#if RAFTINFER_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

}  // namespace raftinfer
```

Create `cpp/src/c_api.cpp`:

```cpp
#include <raftinfer/c_api.h>

#include "engine.hpp"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

struct RaftInferEngineHandle { raftinfer::Engine engine; };
static thread_local std::string g_last_error;

static RaftInferStatus fail(RaftInferStatusCode code, const char* message) {
  g_last_error = message;
  return RaftInferStatus{code, g_last_error.c_str()};
}

extern "C" RaftInferStatus raftinfer_engine_create(
    const RaftInferEngineConfig* config, RaftInferEngineHandle** out_engine) {
  g_last_error.clear();
  if (config == nullptr || out_engine == nullptr) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "config and out_engine are required");
  }
  if (config->struct_size != sizeof(RaftInferEngineConfig)) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "RaftInferEngineConfig size mismatch");
  }
  try {
    auto handle = std::make_unique<RaftInferEngineHandle>(RaftInferEngineHandle{raftinfer::Engine{*config}});
    *out_engine = handle.release();
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::invalid_argument& error) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::exception& error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  }
}

extern "C" void raftinfer_engine_destroy(RaftInferEngineHandle* engine) { delete engine; }

extern "C" int32_t raftinfer_engine_is_cuda_enabled(const RaftInferEngineHandle* engine) {
  return engine != nullptr && engine->engine.cuda_enabled() ? 1 : 0;
}

extern "C" const char* raftinfer_last_error_message(void) { return g_last_error.c_str(); }
```

Add to `cpp/CMakeLists.txt`:

```cmake
target_sources(raftinfer_cpp PRIVATE src/c_api.cpp src/engine.cpp)
target_compile_definitions(raftinfer_cpp PRIVATE RAFTINFER_ENABLE_CUDA=$<BOOL:${RAFTINFER_ENABLE_CUDA}>)
```

- [ ] **Step 5: Run lifecycle tests**

Run:

```bash
cmake --build build/host
ctest --test-dir build/host --output-on-failure
```

Expected: the single C++ test passes and reports no leaked exception across the ABI.

- [ ] **Step 6: Commit the ABI**

```bash
git add cpp/include/raftinfer/c_api.h cpp/src cpp/CMakeLists.txt cpp/tests/c_api_test.cpp
git commit -m "feat: add opaque engine C ABI"
```

---

### Task 3: Rust system crate, safe wrapper, and host CLI

**Files:**
- Create: `Cargo.toml`
- Create: `rust/raftinfer-sys/Cargo.toml`
- Create: `rust/raftinfer-sys/build.rs`
- Create: `rust/raftinfer-sys/src/lib.rs`
- Create: `rust/raftinfer-runtime/Cargo.toml`
- Create: `rust/raftinfer-runtime/src/lib.rs`
- Create: `rust/raftinfer-runtime/tests/engine.rs`
- Create: `rust/raftinfer-cli/Cargo.toml`
- Create: `rust/raftinfer-cli/src/main.rs`

**Interfaces:**
- Consumes: the complete C ABI from Task 2.
- Produces: safe `raftinfer_runtime::Engine`, `EngineConfig`, `Error`, and a `raftinfer-cli info` command.

- [ ] **Step 1: Create the Rust workspace and failing safe-wrapper test**

Create root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["rust/raftinfer-sys", "rust/raftinfer-runtime", "rust/raftinfer-cli"]
```

Create `rust/raftinfer-runtime/tests/engine.rs`:

```rust
use raftinfer_runtime::{Engine, EngineConfig};

#[test]
fn creates_host_engine_and_reports_backend() {
    let engine = Engine::new(EngineConfig::default()).expect("engine creation");
    assert!(!engine.cuda_enabled());
}

#[test]
fn rejects_zero_pool_size() {
    let error = Engine::new(EngineConfig {
        initial_pool_bytes: 0,
        ..EngineConfig::default()
    })
    .expect_err("zero pool must fail");
    assert!(error.to_string().contains("initial_pool_bytes"));
}
```

- [ ] **Step 2: Add crate manifests and verify the test cannot compile**

Create `rust/raftinfer-sys/Cargo.toml`:

```toml
[package]
name = "raftinfer-sys"
version = "0.1.0"
edition = "2024"
links = "raftinfer_cpp"

[build-dependencies]
cmake = "0.1"
```

Create `rust/raftinfer-runtime/Cargo.toml`:

```toml
[package]
name = "raftinfer-runtime"
version = "0.1.0"
edition = "2024"

[dependencies]
raftinfer-sys = { path = "../raftinfer-sys" }
```

Create `rust/raftinfer-cli/Cargo.toml`:

```toml
[package]
name = "raftinfer-cli"
version = "0.1.0"
edition = "2024"

[dependencies]
raftinfer-runtime = { path = "../raftinfer-runtime" }
```

Run `cargo test -p raftinfer-runtime`.

Expected: compile fails because the three crate source files and `raftinfer-sys/build.rs` do not exist.

- [ ] **Step 3: Implement the raw bindings and native build**

Create `rust/raftinfer-sys/build.rs`:

```rust
fn main() {
    let cuda = std::env::var("RAFTINFER_ENABLE_CUDA").unwrap_or_else(|_| "OFF".into());
    let dst = cmake::Config::new("../..")
        .define("RAFTINFER_ENABLE_CUDA", cuda)
        .define("RAFTINFER_BUILD_TESTS", "OFF")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=raftinfer_cpp");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
    println!("cargo:rerun-if-changed=../../cpp");
    println!("cargo:rerun-if-changed=../../CMakeLists.txt");
}
```

The root CMake project must install the static library. Append to `cpp/CMakeLists.txt`:

```cmake
include(GNUInstallDirs)
install(TARGETS raftinfer_cpp ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
```

Create `rust/raftinfer-sys/src/lib.rs`:

```rust
use std::ffi::{c_char, c_int};

#[repr(C)]
pub struct RaftInferEngineHandle { _private: [u8; 0] }

#[repr(C)]
pub struct RaftInferEngineConfig {
    pub struct_size: usize,
    pub device_id: i32,
    pub initial_pool_bytes: u64,
}

#[repr(C)]
pub struct RaftInferStatus {
    pub code: c_int,
    pub message: *const c_char,
}

unsafe extern "C" {
    pub fn raftinfer_engine_create(
        config: *const RaftInferEngineConfig,
        out_engine: *mut *mut RaftInferEngineHandle,
    ) -> RaftInferStatus;
    pub fn raftinfer_engine_destroy(engine: *mut RaftInferEngineHandle);
    pub fn raftinfer_engine_is_cuda_enabled(engine: *const RaftInferEngineHandle) -> i32;
}
```

- [ ] **Step 4: Implement the safe RAII wrapper and CLI**

Create `rust/raftinfer-runtime/src/lib.rs`:

```rust
use std::{ffi::CStr, fmt, ptr::NonNull};

#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub device_id: i32,
    pub initial_pool_bytes: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { device_id: 0, initial_pool_bytes: 64 * 1024 * 1024 }
    }
}

#[derive(Debug)]
pub struct Error { code: i32, message: String }

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RAFTINFER error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
pub struct Engine { raw: NonNull<raftinfer_sys::RaftInferEngineHandle> }

impl Engine {
    pub fn new(config: EngineConfig) -> Result<Self, Error> {
        let native = raftinfer_sys::RaftInferEngineConfig {
            struct_size: std::mem::size_of::<raftinfer_sys::RaftInferEngineConfig>(),
            device_id: config.device_id,
            initial_pool_bytes: config.initial_pool_bytes,
        };
        let mut raw = std::ptr::null_mut();
        let status = unsafe { raftinfer_sys::raftinfer_engine_create(&native, &mut raw) };
        if status.code != 0 {
            let message = if status.message.is_null() {
                "unknown native error".to_owned()
            } else {
                unsafe { CStr::from_ptr(status.message) }.to_string_lossy().into_owned()
            };
            return Err(Error { code: status.code, message });
        }
        Ok(Self { raw: NonNull::new(raw).expect("successful create returned null") })
    }

    pub fn cuda_enabled(&self) -> bool {
        unsafe { raftinfer_sys::raftinfer_engine_is_cuda_enabled(self.raw.as_ptr()) != 0 }
    }
}

impl Drop for Engine {
    fn drop(&mut self) { unsafe { raftinfer_sys::raftinfer_engine_destroy(self.raw.as_ptr()) } }
}
```

Create `rust/raftinfer-cli/src/main.rs`:

```rust
use raftinfer_runtime::{Engine, EngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command = std::env::args().nth(1).unwrap_or_else(|| "info".into());
    if command != "info" { return Err(format!("unknown command: {command}").into()); }
    let engine = Engine::new(EngineConfig::default())?;
    println!("backend={}", if engine.cuda_enabled() { "cuda" } else { "host" });
    Ok(())
}
```

- [ ] **Step 5: Run Rust and C++ host checks**

Run:

```bash
cargo fmt --check
cargo test --workspace
cargo run -p raftinfer-cli -- info
```

Expected: all Rust tests pass and the CLI prints `backend=host`.

- [ ] **Step 6: Commit the Rust boundary**

```bash
git add Cargo.toml Cargo.lock rust cpp/CMakeLists.txt
git commit -m "feat: add safe Rust engine wrapper"
```

---

### Task 4: RAFT/RMM device context and custom CUDA smoke kernel

**Files:**
- Create: `cpp/foundation/device_context.hpp`
- Create: `cpp/foundation/device_context.cu`
- Create: `cpp/kernels/smoke.cu`
- Create: `cpp/tools/raftinfer_smoke.cpp`
- Modify: `cpp/src/engine.hpp`
- Modify: `cpp/src/engine.cpp`
- Modify: `cpp/include/raftinfer/c_api.h`
- Modify: `cpp/src/c_api.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Consumes: opaque engine ABI and RMM pool-size configuration.
- Produces: `raftinfer_engine_run_smoke`, `RaftInferSmokeResult`, `DeviceContext::run_smoke`, and the `raftinfer-smoke` executable.

- [ ] **Step 1: Add a failing CUDA smoke ABI test to the tool**

Create `cpp/tools/raftinfer_smoke.cpp`:

```cpp
#include <raftinfer/c_api.h>

#include <iostream>

int main() {
  RaftInferEngineConfig config{sizeof(RaftInferEngineConfig), 0, 64U * 1024U * 1024U};
  RaftInferEngineHandle* engine = nullptr;
  RaftInferStatus status = raftinfer_engine_create(&config, &engine);
  if (status.code != RAFTINFER_STATUS_OK) {
    std::cerr << status.message << '\n';
    return 1;
  }
  RaftInferSmokeResult result{};
  status = raftinfer_engine_run_smoke(engine, &result);
  raftinfer_engine_destroy(engine);
  if (status.code != RAFTINFER_STATUS_OK) {
    std::cerr << status.message << '\n';
    return 1;
  }
  std::cout << "{\"device_id\":" << result.device_id
            << ",\"element_count\":" << result.element_count
            << ",\"checksum\":" << result.checksum << "}\n";
  return result.checksum == 523776 ? 0 : 2;
}
```

- [ ] **Step 2: Configure the CUDA build and verify missing symbols fail**

Inside the RAPIDS container, run:

```bash
cmake -S . -B build/gpu -G Ninja -DRAFTINFER_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/gpu --target raftinfer-smoke
```

Expected: compile fails because `RaftInferSmokeResult` and `raftinfer_engine_run_smoke` are not defined.

- [ ] **Step 3: Define DeviceContext with RAFT 26.06 and RMM 26.06 value resources**

Create `cpp/foundation/device_context.hpp`:

```cpp
#pragma once

#include <raftinfer/c_api.h>
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cstdint>

namespace raftinfer {

class DeviceContext {
 public:
  DeviceContext(int device_id, uint64_t initial_pool_bytes);
  RaftInferSmokeResult run_smoke();

 private:
  static int select_device(int device_id);
  int device_id_;
  raft::device_resources resources_;
  rmm::mr::cuda_memory_resource cuda_resource_;
  rmm::mr::pool_memory_resource pool_;
};

}  // namespace raftinfer
```

Create `cpp/foundation/device_context.cu` with a stream-ordered allocation, launch, copy, synchronization, and deallocation:

```cpp
#include "device_context.hpp"

#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <numeric>
#include <stdexcept>
#include <vector>

extern void raftinfer_launch_smoke(uint32_t* values, uint32_t count, cudaStream_t stream);

namespace raftinfer {

int DeviceContext::select_device(int device_id) {
  cudaError_t error = cudaSetDevice(device_id);
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  return device_id;
}

DeviceContext::DeviceContext(int device_id, uint64_t initial_pool_bytes)
    : device_id_(select_device(device_id)),
      resources_(),
      cuda_resource_(),
      pool_(cuda_resource_, initial_pool_bytes) {}

RaftInferSmokeResult DeviceContext::run_smoke() {
  constexpr uint32_t count = 1024;
  auto stream_view = raft::resource::get_cuda_stream(resources_);
  cudaStream_t stream = stream_view.value();
  auto stream_ref = cuda::stream_ref{stream};
  auto resource = rmm::device_async_resource_ref{pool_};
  auto* device = static_cast<uint32_t*>(resource.allocate(stream_ref, count * sizeof(uint32_t)));
  raftinfer_launch_smoke(device, count, stream);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    resource.deallocate(stream_ref, device, count * sizeof(uint32_t));
    throw std::runtime_error(cudaGetErrorString(error));
  }
  std::vector<uint32_t> host(count);
  error = cudaMemcpyAsync(
      host.data(), device, count * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  resource.deallocate(stream_ref, device, count * sizeof(uint32_t));
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  uint64_t checksum = std::accumulate(host.begin(), host.end(), uint64_t{0});
  return RaftInferSmokeResult{device_id_, count, checksum};
}

}  // namespace raftinfer
```

- [ ] **Step 4: Implement the custom CUDA kernel and ABI method**

Create `cpp/kernels/smoke.cu`:

```cpp
#include <cuda_runtime.h>
#include <cstdint>

__global__ void raftinfer_smoke_kernel(uint32_t* values, uint32_t count) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) values[index] = index;
}

void raftinfer_launch_smoke(uint32_t* values, uint32_t count, cudaStream_t stream) {
  raftinfer_smoke_kernel<<<(count + 255) / 256, 256, 0, stream>>>(values, count);
}
```

Add to `cpp/include/raftinfer/c_api.h`:

```cpp
typedef struct RaftInferSmokeResult {
  int32_t device_id;
  uint32_t element_count;
  uint64_t checksum;
} RaftInferSmokeResult;

RaftInferStatus raftinfer_engine_run_smoke(RaftInferEngineHandle* engine, RaftInferSmokeResult* out_result);
```

Replace `cpp/src/engine.hpp` with:

```cpp
#pragma once

#include <raftinfer/c_api.h>
#include <memory>

namespace raftinfer {

#if RAFTINFER_ENABLE_CUDA
class DeviceContext;
#endif

class Engine {
 public:
  explicit Engine(const RaftInferEngineConfig& config);
  ~Engine();
  bool cuda_enabled() const noexcept;
  RaftInferSmokeResult run_smoke();

 private:
  int device_id_;
  uint64_t initial_pool_bytes_;
#if RAFTINFER_ENABLE_CUDA
  std::unique_ptr<DeviceContext> device_;
#endif
};

}  // namespace raftinfer
```

Replace `cpp/src/engine.cpp` with:

```cpp
#include "engine.hpp"

#if RAFTINFER_ENABLE_CUDA
#include "../foundation/device_context.hpp"
#endif

#include <stdexcept>

namespace raftinfer {

Engine::Engine(const RaftInferEngineConfig& config)
    : device_id_(config.device_id), initial_pool_bytes_(config.initial_pool_bytes) {
  if (device_id_ < 0) throw std::invalid_argument("device_id must be non-negative");
  if (initial_pool_bytes_ == 0) throw std::invalid_argument("initial_pool_bytes must be non-zero");
#if RAFTINFER_ENABLE_CUDA
  device_ = std::make_unique<DeviceContext>(device_id_, initial_pool_bytes_);
#endif
}

Engine::~Engine() = default;

bool Engine::cuda_enabled() const noexcept {
#if RAFTINFER_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

RaftInferSmokeResult Engine::run_smoke() {
#if RAFTINFER_ENABLE_CUDA
  return device_->run_smoke();
#else
  throw std::runtime_error("CUDA backend is not enabled");
#endif
}

}  // namespace raftinfer
```

Add to `cpp/src/c_api.cpp`:

```cpp
extern "C" RaftInferStatus raftinfer_engine_run_smoke(
    RaftInferEngineHandle* engine, RaftInferSmokeResult* out_result) {
  g_last_error.clear();
  if (engine == nullptr || out_result == nullptr) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "engine and out_result are required");
  }
  if (!engine->engine.cuda_enabled()) {
    return fail(RAFTINFER_STATUS_UNAVAILABLE, "CUDA backend is not enabled");
  }
  try {
    *out_result = engine->engine.run_smoke();
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::exception& error) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, error.what());
  }
}
```

Update `cpp/CMakeLists.txt`:

```cmake
if(RAFTINFER_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(raft 26.06 CONFIG REQUIRED)
  find_package(rmm 26.06 CONFIG REQUIRED)
  target_sources(raftinfer_cpp PRIVATE foundation/device_context.cu kernels/smoke.cu)
  target_link_libraries(raftinfer_cpp PUBLIC raft::raft rmm::rmm CUDA::cudart)
  target_compile_options(raftinfer_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
endif()

add_executable(raftinfer-smoke tools/raftinfer_smoke.cpp)
target_link_libraries(raftinfer-smoke PRIVATE raftinfer_cpp)
```

- [ ] **Step 5: Build and run on an idle RTX 5090**

Run inside the GPU container:

```bash
cmake -S . -B build/gpu -G Ninja -DRAFTINFER_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/gpu --target raftinfer-smoke
./build/gpu/cpp/raftinfer-smoke
```

Expected output:

```json
{"device_id":0,"element_count":1024,"checksum":523776}
```

- [ ] **Step 6: Commit the GPU foundation**

```bash
git add cpp/foundation cpp/kernels/smoke.cu cpp/tools/raftinfer_smoke.cpp cpp/include/raftinfer/c_api.h cpp/src cpp/CMakeLists.txt
git commit -m "feat: add RAFT RMM GPU smoke context"
```

---

### Task 5: Reproducible RAPIDS container and shared-GPU preflight

**Files:**
- Create: `containers/Dockerfile.dev`
- Create: `scripts/gpu-preflight.sh`
- Create: `scripts/gpu-smoke.sh`
- Create: `scripts/local-check.sh`
- Create: `scripts/sync-target.sh`
- Create: `tests/golden/smoke_result.json`

**Interfaces:**
- Consumes: `raftinfer-smoke` from Task 4.
- Produces: `raftinfer-dev:26.06-cuda13`, a no-side-effect preflight command, and repeatable local/remote verification scripts.

- [ ] **Step 1: Write the shared-GPU preflight script and its shell syntax test**

Create `scripts/gpu-preflight.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

minimum_free_mib="${RAFTINFER_MIN_FREE_MIB:-2048}"
compute_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
gpu_row="$(nvidia-smi --query-gpu=memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | head -n 1)"
IFS=',' read -r free_mib utilization temperature <<<"${gpu_row}"
free_mib="${free_mib// /}"
utilization="${utilization// /}"
temperature="${temperature// /}"

if [[ -n "${compute_apps}" ]]; then
  echo "RAFTINFER GPU preflight refused: active compute applications detected" >&2
  echo "${compute_apps}" >&2
  exit 20
fi
if (( free_mib < minimum_free_mib )); then
  echo "RAFTINFER GPU preflight refused: free=${free_mib}MiB required=${minimum_free_mib}MiB" >&2
  exit 21
fi
if (( utilization > 5 )); then
  echo "RAFTINFER GPU preflight refused: utilization=${utilization}%" >&2
  exit 22
fi

printf 'gpu_preflight=ok free_mib=%s utilization=%s temperature=%s\n' \
  "${free_mib}" "${utilization}" "${temperature}"
```

Run `bash -n scripts/gpu-preflight.sh`.

Expected: exit code 0 with no output.

- [ ] **Step 2: Create the pinned development container**

Create `containers/Dockerfile.dev`:

```dockerfile
FROM rapidsai/base:26.06-cuda13-py3.13

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates ninja-build pkg-config \
 && rm -rf /var/lib/apt/lists/*

USER rapids
ENV RUSTUP_HOME=/home/rapids/.rustup
ENV CARGO_HOME=/home/rapids/.cargo
ENV PATH=/home/rapids/.cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
 | sh -s -- -y --profile minimal --default-toolchain 1.96.0

WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
```

Build with:

```bash
docker build -f containers/Dockerfile.dev -t raftinfer-dev:26.06-cuda13 .
```

Expected: image build completes and `docker run --rm raftinfer-dev:26.06-cuda13 rustc --version` prints Rust 1.96.0.

- [ ] **Step 3: Add local and GPU runners**

Create `scripts/local-check.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cmake -S . -B build/host -G Ninja -DRAFTINFER_ENABLE_CUDA=OFF
cmake --build build/host
ctest --test-dir build/host --output-on-failure
cargo fmt --check
cargo test --workspace
```

Create `scripts/gpu-smoke.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${repo_root}/scripts/gpu-preflight.sh"
docker run --rm --gpus all \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${repo_root}:/workspace" -w /workspace \
  raftinfer-dev:26.06-cuda13 \
  bash -lc 'cmake -S . -B build/gpu -G Ninja -DRAFTINFER_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build/gpu --target raftinfer-smoke && ./build/gpu/cpp/raftinfer-smoke'
```

Create `scripts/sync-target.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
target="${RAFTINFER_TARGET:-<validation-root>}"
destination="${RAFTINFER_TARGET_DIR:-<repo>}"
ssh "${target}" "mkdir -p '${destination}'"
rsync -a \
  --exclude .git --exclude build --exclude target --exclude evidence/local \
  ./ "${target}:${destination}/"
printf 'synced_target=%s synced_dir=%s\n' "${target}" "${destination}"
```

Create `tests/golden/smoke_result.json`:

```json
{"device_id":0,"element_count":1024,"checksum":523776}
```

Mark scripts executable with `chmod +x scripts/*.sh`.

- [ ] **Step 4: Run host checks and target preflight**

Local command:

```bash
scripts/local-check.sh
```

Sync and run the remote preflight in the project-specific target directory:

```bash
scripts/sync-target.sh
ssh <validation-root> 'cd <repo> && scripts/gpu-preflight.sh'
```

Expected on the currently observed idle target: a single `gpu_preflight=ok` line. If another process is using the GPU, exit 20, 21, or 22 is the correct result and the GPU smoke is deferred.

- [ ] **Step 5: Run the containerized GPU smoke and compare the golden result**

Run on the target after `scripts/sync-target.sh`:

```bash
ssh <validation-root> 'cd <repo> && scripts/gpu-smoke.sh' | tee /tmp/raftinfer-smoke-output.txt
tail -n 1 /tmp/raftinfer-smoke-output.txt | diff -u tests/golden/smoke_result.json -
```

Expected: `diff` exits 0. Re-run `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` and confirm the RAFTINFER container is gone.

- [ ] **Step 6: Commit the reproducible runners**

```bash
git add containers scripts tests/golden/smoke_result.json
git commit -m "build: add reproducible GPU smoke environment"
```

---

### Task 6: Dependency provenance and M0 verification record

**Files:**
- Create: `docs/provenance/dependencies.md`
- Create: `docs/verification/m0.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: successful host and GPU evidence from Tasks 1-5.
- Produces: auditable dependency, environment, safety, and smoke evidence for the M0 completion claim.

- [ ] **Step 1: Write dependency provenance**

Create `docs/provenance/dependencies.md` with this table and the exact container digest reported by `docker image inspect`:

```markdown
# M0 Dependency Provenance

| Component | Version/source | License | M0 use |
|---|---|---|---|
| RAPIDS base image | `rapidsai/base:26.06-cuda13-py3.13` plus recorded image digest | NVIDIA/RAPIDS image terms | Reproducible CUDA/RAFT/RMM environment |
| RAFT | 26.06 from the RAPIDS image | Apache-2.0 | `raft::device_resources` and stream ownership |
| RMM | 26.06 from the RAPIDS image | Apache-2.0 | CUDA memory resource and pool |
| CUDA Toolkit | CUDA 13 major from the RAPIDS image | NVIDIA CUDA Toolkit EULA | `sm_120a` compilation/runtime |
| Rust | 1.96.0 via rustup | MIT OR Apache-2.0 | Safe runtime and CLI |
| cmake crate | Resolved version in `Cargo.lock` | MIT OR Apache-2.0 | Native CMake build from Cargo |

The project does not vendor or import `bw24` code in M0.
```

- [ ] **Step 2: Write the verification record from actual outputs**

Create `docs/verification/m0.md` containing:

```markdown
# M0 Verification

## Host-only

- Command: `scripts/local-check.sh`
- Result: PASS
- Platform: macOS arm64
- CUDA backend: disabled

## Target GPU

- Host: `<validation-root>`
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32607 MiB
- Driver: 580.159.03
- Safety preflight: PASS with no active compute applications
- Command: `scripts/gpu-smoke.sh`
- Expected result: `{"device_id":0,"element_count":1024,"checksum":523776}`
- Result: PASS only after the observed output matches the expected result byte-for-byte

## Scope

This milestone validates the build, ABI, RAFT/RMM ownership, RMM allocation,
custom CUDA launch, result copy, and Rust-to-C++ lifecycle. It does not validate
model loading, LLM operators, CUDA Graph, or inference performance.
```

Do not mark the target result PASS until the remote command has actually succeeded. If shared-GPU preflight blocks execution, record `DEFERRED: shared GPU busy` and keep M0 incomplete.

- [ ] **Step 3: Add a concise README**

Create `README.md`:

````markdown
# RAFTInfer

An RTX 50-series LLM inference runtime using RAFT/RMM as the GPU foundation and
custom CUDA/CUTLASS kernels for performance-critical operators.

Current milestone: M0 full-stack smoke. No model inference is implemented yet.

## Host checks

```bash
scripts/local-check.sh
```

## RTX 50 GPU smoke

```bash
docker build -f containers/Dockerfile.dev -t raftinfer-dev:26.06-cuda13 .
scripts/gpu-smoke.sh
```

The GPU runner refuses to start when it detects another compute workload or
insufficient memory headroom. See the approved design under
`docs/superpowers/specs/` for scope and safety rules.
````

- [ ] **Step 4: Run the complete M0 verification sequence**

Run locally:

```bash
scripts/local-check.sh
git diff --check
```

Run remotely only after preflight passes:

```bash
scripts/gpu-smoke.sh
```

Expected: host checks pass, GPU JSON matches the golden file, no RAFTINFER process remains on the GPU, and the working tree contains only the documentation changes for this task.

- [ ] **Step 5: Commit the M0 record**

```bash
git add README.md docs/provenance/dependencies.md docs/verification/m0.md
git commit -m "docs: record M0 foundation verification"
```

## M0 completion gate

M0 is complete only when all six task commits exist, `scripts/local-check.sh` passes on the local host, the shared-GPU preflight passes during an uncontended window, `scripts/gpu-smoke.sh` returns the exact golden JSON on the RTX 5090, the container exits without leaving a GPU process, and `docs/verification/m0.md` records actual rather than anticipated evidence.
