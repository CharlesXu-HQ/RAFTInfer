#include "device_context.hpp"

#include "../execution/execution_context.hpp"
#include "../execution/workspace_arena.hpp"

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

extern void brt_launch_smoke(uint32_t* values, uint32_t count, cudaStream_t stream);

namespace brt {

namespace {

[[noreturn]] void throw_cuda_error(cudaError_t error) {
  throw std::runtime_error(cudaGetErrorString(error));
}

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id) {
    cudaError_t error = cudaGetDevice(&previous_device_);
    if (error != cudaSuccess) throw_cuda_error(error);
    error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
      const cudaError_t selection_error = error;
      const cudaError_t restoration_error = cudaSetDevice(previous_device_);
      if (restoration_error != cudaSuccess) throw_cuda_error(restoration_error);
      throw_cuda_error(selection_error);
    }
    restore_device_ = true;
  }

  void restore() {
    if (!restore_device_) return;
    const cudaError_t error = cudaSetDevice(previous_device_);
    if (error != cudaSuccess) throw_cuda_error(error);
    restore_device_ = false;
  }

  ~DeviceGuard() noexcept {
    if (restore_device_) (void)cudaSetDevice(previous_device_);
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_device_{};
  bool restore_device_{};
};

}  // namespace

class DeviceContext::Resources {
 public:
  Resources(int device_id, uint64_t initial_pool_bytes)
      : cuda_resource_(), pool_(cuda_resource_, initial_pool_bytes), workspace_arena_(), resources_() {
    raft::resource::set_workspace_resource(resources_, raft::mr::device_resource{pool_});
    raft::resource::set_large_workspace_resource(resources_, raft::mr::device_resource{pool_});
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDeviceProperties(&properties, device_id);
    if (error != cudaSuccess) throw_cuda_error(error);
    compute_capability_major_ = properties.major;
    compute_capability_minor_ = properties.minor;
    max_shared_memory_per_block_ = properties.sharedMemPerBlock;
    auto stream_view = raft::resource::get_cuda_stream(resources_);
    cudaStream_t stream = stream_view.value();
    workspace_arena_.emplace(
        rmm::device_async_resource_ref{pool_}, cuda::stream_ref{stream}, stream, kWorkspaceBytes);
  }

  ExecutionContext execution_context(int device_id) {
    WorkspaceArena& workspace = *workspace_arena_;
    auto stream_view = raft::resource::get_cuda_stream(resources_);
    cudaStream_t stream = stream_view.value();
    return ExecutionContext{
        resources_,
        rmm::device_async_resource_ref{pool_},
        stream,
        workspace,
        device_id,
        compute_capability_major_,
        compute_capability_minor_,
        max_shared_memory_per_block_};
  }

  void probe_workspace() {
    WorkspaceArena& workspace = *workspace_arena_;
    workspace.reset();
    (void)workspace.allocate(1, alignof(uint32_t));
    workspace.reset();
  }

  static constexpr std::size_t kWorkspaceBytes = 1024 * 1024;
  rmm::mr::cuda_memory_resource cuda_resource_;
  rmm::mr::pool_memory_resource pool_;
  std::optional<WorkspaceArena> workspace_arena_;
  raft::device_resources resources_;
  int compute_capability_major_{};
  int compute_capability_minor_{};
  int max_shared_memory_per_block_{};
};

DeviceContext::DeviceContext(int device_id, uint64_t initial_pool_bytes) : device_id_(device_id) {
  DeviceGuard device_guard{device_id_};
  try {
    resources_ = std::make_unique<Resources>(device_id_, initial_pool_bytes);
    device_guard.restore();
  } catch (...) {
    // If the caller's device cannot be restored, do not let RAFT/RMM
    // destructors run under an unknown device. The allocation is deliberately
    // leaked; construction reports the CUDA failure through the C ABI.
    (void)resources_.release();
    throw;
  }
}

DeviceContext::~DeviceContext() noexcept {
  try {
    DeviceGuard device_guard{device_id_};
    resources_.reset();
    device_guard.restore();
  } catch (...) {
    // A failed CUDA device query/selection cannot safely preserve the caller's
    // current device. If resources remain, deliberately leak them rather than
    // invoke RAFT/RMM destructors under an unknown device. Teardown is
    // best-effort and must never let an exception cross the C ABI.
    (void)resources_.release();
  }
}

BrtSmokeResult DeviceContext::run_smoke() {
  constexpr uint32_t count = 1024;
  DeviceGuard device_guard{device_id_};
  ExecutionContext execution_context = resources_->execution_context(device_id_);
  WorkspaceArena& workspace = execution_context.workspace();
  resources_->probe_workspace();
  uint32_t* device_values =
      static_cast<uint32_t*>(workspace.allocate(count * sizeof(uint32_t), alignof(uint32_t)));
  brt_launch_smoke(device_values, count, execution_context.stream());
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    const std::string message = cudaGetErrorString(error);
    (void)cudaStreamSynchronize(execution_context.stream());
    workspace.reset();
    throw std::runtime_error(message);
  }
  std::array<uint32_t, count> host{};
  error = cudaMemcpyAsync(
      host.data(), device_values, count * sizeof(uint32_t), cudaMemcpyDeviceToHost, execution_context.stream());
  const cudaError_t synchronization_error = cudaStreamSynchronize(execution_context.stream());
  if (error == cudaSuccess) error = synchronization_error;
  if (error != cudaSuccess) {
    const std::string message = cudaGetErrorString(error);
    workspace.reset();
    throw std::runtime_error(message);
  }
  workspace.reset();
  uint64_t checksum = std::accumulate(host.begin(), host.end(), uint64_t{0});
  BrtSmokeResult result{device_id_, count, checksum};
  device_guard.restore();
  return result;
}

}  // namespace brt
