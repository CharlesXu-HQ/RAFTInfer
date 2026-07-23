# M0 Dependency Provenance

M0 is a RAFT/RMM foundation milestone with a custom CUDA smoke kernel. It does
not vendor or import `bw24` code.

## Runtime and build components

| Component | Version/source | License | M0 use |
|---|---|---|---|
| RAPIDS base image | `rapidsai/base:26.06-cuda13-py3.13`; recorded base image ID `sha256:95b361b2f2e67489a4eb53f03e32a315be9272b169b5e2f161bfc9e576e1aaf6` | NVIDIA/RAPIDS image terms | Reproducible CUDA/RAFT/RMM environment |
| BRT dev image | `brt-dev:26.06-cuda13`; recorded final image ID `sha256:9e7ce1e85752d43a53cc2e8bf5c44f4ec610d43d38b28d922a4fab9701c6c340`; final user `rapids` | Project image assembled from listed dependencies | Target M0 CUDA build and smoke execution |
| RAFT | 26.06 from the RAPIDS image | Apache-2.0 | `raft::device_resources` and stream/resource ownership |
| RMM | 26.06 from the RAPIDS image | Apache-2.0 | CUDA memory resource, pool setup, and device allocation |
| CUDA Toolkit | CUDA 13.2.78 from the RAPIDS image | NVIDIA CUDA Toolkit EULA | `sm_120a` compilation/runtime for RTX 50 |
| CMake executable | 3.30.4 installed with `/opt/conda/bin/python -m pip install cmake==3.30.4` | BSD-3-Clause for CMake; PyPI wheel packaging terms apply | RAFT/RMM-compatible native CMake configure/build |
| Rust | 1.96.0 via Rustup | MIT OR Apache-2.0 | Safe runtime crates and CLI |
| `cmake` Rust crate | `0.1.58` from `registry+https://github.com/rust-lang/crates.io-index`, checksum `c0f78a02292a74a88ac736019ab962ece0bc380e3f977bf72e376c5d78ff0678` | MIT OR Apache-2.0 | `brt-sys` build script invokes the native CMake build |
| `cc` Rust crate | `1.3.0` from `registry+https://github.com/rust-lang/crates.io-index`, checksum `c89588d05638b5b4594a3348a2d6c20277e43a7f5c5202b05cc56888475a47b8` | MIT OR Apache-2.0 | Transitive build dependency of the `cmake` crate |
| G++ host compiler | G++ 13.3 installed from the base image apt repositories | GPL-3.0-or-later with GCC Runtime Library Exception | Host compiler for CMake/NVCC CUDA builds |

## Endpoint and reproducibility notes

- `containers/Dockerfile.dev` keeps official endpoint defaults:
  `CMAKE_PIP_INDEX_URL=https://pypi.org/simple`,
  `RUSTUP_DIST_SERVER=https://static.rust-lang.org`, and
  `RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup`.
- The successful target build used
  `CMAKE_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`,
  `RUSTUP_DIST_SERVER=https://rsproxy.cn`, and
  `RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup` because the official Rustup
  download stalled on the target network.
- Apt repositories, Rustup downloads, and the PyPI CMake wheel fetch are mutable
  provenance risks. The CMake wheel is version-pinned but not hash-pinned.
- The project should record upstream commit/path/license/local changes before
  reusing any external kernel implementation.

## `bw24` reuse rule

M0 contains no `bw24` source code. Future `bw24` custom operators may be reused
directly only when all of the following are recorded:

1. The operator is already performance-optimized for the target class of GPU or
   has target RTX 50 performance evidence.
2. License and provenance allow reuse in this Apache-2.0 project.
3. Functional and numerical correctness match the BRT kernel contract.
4. Upstream commit, source path, license, imported files, and local
   modifications are documented.
