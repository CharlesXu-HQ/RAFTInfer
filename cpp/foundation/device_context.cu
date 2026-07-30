#include "device_context.hpp"

#include "../execution/execution_context.hpp"
#include "../execution/workspace_arena.hpp"
#include "../model/cuda_weights.hpp"
#include "../model/model.hpp"

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

extern void raftinfer_launch_smoke(uint32_t* values, uint32_t count, cudaStream_t stream);

namespace raftinfer {

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
      : cuda_resource_(), pool_(cuda_resource_, initial_pool_bytes),
        statistics_(pool_), workspace_arena_(), resources_() {
    raft::resource::set_workspace_resource(
        resources_, raft::mr::device_resource{statistics_});
    raft::resource::set_large_workspace_resource(
        resources_, raft::mr::device_resource{statistics_});
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDeviceProperties(&properties, device_id);
    if (error != cudaSuccess) throw_cuda_error(error);
    compute_capability_major_ = properties.major;
    compute_capability_minor_ = properties.minor;
    max_shared_memory_per_block_ = properties.sharedMemPerBlock;
    auto stream_view = raft::resource::get_cuda_stream(resources_);
    cudaStream_t stream = stream_view.value();
    workspace_arena_.emplace(
        rmm::device_async_resource_ref{statistics_}, cuda::stream_ref{stream},
        stream, kWorkspaceBytes);
  }

  ExecutionContext execution_context(int device_id) {
    WorkspaceArena& workspace = *workspace_arena_;
    return execution_context(device_id, workspace);
  }

  ExecutionContext execution_context(int device_id, WorkspaceArena& workspace) {
    auto stream_view = raft::resource::get_cuda_stream(resources_);
    cudaStream_t stream = stream_view.value();
    return ExecutionContext{
        resources_,
        rmm::device_async_resource_ref{statistics_},
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

  std::uint64_t peak_allocated_bytes() const noexcept {
    return static_cast<std::uint64_t>(
        statistics_.get_bytes_counter().peak);
  }

  static constexpr std::size_t kWorkspaceBytes = 1024 * 1024;
  rmm::mr::cuda_memory_resource cuda_resource_;
  rmm::mr::pool_memory_resource pool_;
  rmm::mr::statistics_resource_adaptor statistics_;
  std::optional<WorkspaceArena> workspace_arena_;
  raft::device_resources resources_;
  int compute_capability_major_{};
  int compute_capability_minor_{};
  int max_shared_memory_per_block_{};
};

class DeviceExecutionOwner::Impl {
 public:
  Impl(std::shared_ptr<void> resource_anchor,
       raft::device_resources& resources,
       rmm::device_async_resource_ref memory_resource,
       cudaStream_t stream,
       int device_id,
       int compute_capability_major,
       int compute_capability_minor,
       int max_shared_memory_per_block,
       std::size_t workspace_bytes)
      : resource_anchor_(std::move(resource_anchor)),
        resources_(&resources),
        memory_resource_(memory_resource),
        stream_(stream),
        device_id_(device_id),
        compute_capability_major_(compute_capability_major),
        compute_capability_minor_(compute_capability_minor),
        max_shared_memory_per_block_(max_shared_memory_per_block),
        workspace_(memory_resource_, cuda::stream_ref{stream_}, stream_,
                   workspace_bytes) {}

  ExecutionContext execution_context() {
    return ExecutionContext{
        *resources_,
        memory_resource_,
        stream_,
        workspace_,
        device_id_,
        compute_capability_major_,
        compute_capability_minor_,
        max_shared_memory_per_block_};
  }

  std::shared_ptr<void> resource_anchor_;
  raft::device_resources* resources_;
  rmm::device_async_resource_ref memory_resource_;
  cudaStream_t stream_{};
  int device_id_{};
  int compute_capability_major_{};
  int compute_capability_minor_{};
  int max_shared_memory_per_block_{};
  WorkspaceArena workspace_;
};

DeviceContext::DeviceContext(int device_id, uint64_t initial_pool_bytes) : device_id_(device_id) {
  DeviceGuard device_guard{device_id_};
  resources_ = std::shared_ptr<Resources>(
      new Resources(device_id_, initial_pool_bytes),
      [device_id = device_id_](Resources* resources) noexcept {
        if (resources == nullptr) return;
        try {
          DeviceGuard device_guard{device_id};
          delete resources;
          device_guard.restore();
        } catch (...) {
          // If the original device cannot be selected/restored, do not run
          // RAFT/RMM destructors under an unknown device. This is the same
          // safety policy used by DeviceContext teardown.
        }
      });
  device_guard.restore();
}

DeviceContext::~DeviceContext() noexcept {
  resources_.reset();
}

DeviceExecutionOwner::DeviceExecutionOwner(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

DeviceExecutionOwner::~DeviceExecutionOwner() noexcept {
  if (impl_ == nullptr) return;
  try {
    DeviceGuard device_guard{impl_->device_id_};
    impl_.reset();
    device_guard.restore();
  } catch (...) {
    // Avoid running CUDA/RMM destructors under the wrong device when device
    // selection or restoration fails. The allocation is deliberately leaked.
    (void)impl_.release();
  }
}

ExecutionContext DeviceExecutionOwner::execution_context() {
  if (impl_ == nullptr) {
    throw std::logic_error("device execution owner is not initialized");
  }
  return impl_->execution_context();
}

std::size_t DeviceExecutionOwner::workspace_bytes() const noexcept {
  return impl_ == nullptr ? 0 : impl_->workspace_.capacity();
}

int DeviceExecutionOwner::device_id() const noexcept {
  return impl_ == nullptr ? -1 : impl_->device_id_;
}

std::unique_ptr<DeviceExecutionOwner>
DeviceContext::create_execution_owner(std::size_t workspace_bytes) const {
  if (workspace_bytes == 0) {
    throw std::invalid_argument(
        "device execution workspace_bytes must be non-zero");
  }
  DeviceGuard device_guard{device_id_};
  auto stream_view = raft::resource::get_cuda_stream(resources_->resources_);
  cudaStream_t stream = stream_view.value();
  auto impl = std::make_unique<DeviceExecutionOwner::Impl>(
      std::static_pointer_cast<void>(resources_),
      resources_->resources_,
      rmm::device_async_resource_ref{resources_->statistics_},
      stream,
      device_id_,
      resources_->compute_capability_major_,
      resources_->compute_capability_minor_,
      resources_->max_shared_memory_per_block_,
      workspace_bytes);
  auto owner = std::unique_ptr<DeviceExecutionOwner>(
      new DeviceExecutionOwner(std::move(impl)));
  device_guard.restore();
  return owner;
}

std::uint64_t DeviceContext::peak_allocated_bytes() {
  DeviceGuard device_guard{device_id_};
  const auto peak = resources_->peak_allocated_bytes();
  device_guard.restore();
  return peak;
}

RaftInferSmokeResult DeviceContext::run_smoke() {
  constexpr uint32_t count = 1024;
  DeviceGuard device_guard{device_id_};
  ExecutionContext execution_context = resources_->execution_context(device_id_);
  WorkspaceArena& workspace = execution_context.workspace();
  resources_->probe_workspace();
  uint32_t* device_values =
      static_cast<uint32_t*>(workspace.allocate(count * sizeof(uint32_t), alignof(uint32_t)));
  raftinfer_launch_smoke(device_values, count, execution_context.stream());
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
  RaftInferSmokeResult result{device_id_, count, checksum};
  device_guard.restore();
  return result;
}

std::unique_ptr<model::CudaWeightPlan>
DeviceContext::upload_qwen35_weights(const model::Model& model) {
  return upload_qwen35_weights_for_tests(model, model.qwen35_manifest());
}

std::unique_ptr<model::CudaWeightPlan>
DeviceContext::upload_qwen35_weights_for_tests(
    const model::Model& model, const model::Qwen35Manifest& manifest) {
  DeviceGuard device_guard{device_id_};
  ExecutionContext execution_context = resources_->execution_context(device_id_);
  auto plan = model::CudaWeightPlan::upload(
      execution_context, model, manifest,
      std::static_pointer_cast<void>(resources_));
  device_guard.restore();
  return plan;
}

}  // namespace raftinfer
