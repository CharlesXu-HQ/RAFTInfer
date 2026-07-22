#include "device_context.hpp"

#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

class DeviceAllocation {
 public:
  DeviceAllocation(
      rmm::device_async_resource_ref resource,
      cuda::stream_ref stream_ref,
      cudaStream_t stream,
      std::size_t bytes)
      : resource_(resource),
        stream_ref_(stream_ref),
        stream_(stream),
        bytes_(bytes),
        values_(static_cast<uint32_t*>(resource_.allocate(stream_ref_, bytes_))) {}

  ~DeviceAllocation() noexcept {
    if (values_ != nullptr) {
      (void)cudaStreamSynchronize(stream_);
      try {
        deallocate();
      } catch (...) {
      }
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  uint32_t* data() const noexcept { return values_; }
  cudaError_t synchronize() const noexcept { return cudaStreamSynchronize(stream_); }

  void deallocate() {
    uint32_t* values = std::exchange(values_, nullptr);
    if (values != nullptr) resource_.deallocate(stream_ref_, values, bytes_);
  }

 private:
  rmm::device_async_resource_ref resource_;
  cuda::stream_ref stream_ref_;
  cudaStream_t stream_;
  std::size_t bytes_;
  uint32_t* values_{};
};

}  // namespace

class DeviceContext::Resources {
 public:
  explicit Resources(uint64_t initial_pool_bytes)
      : resources_(), cuda_resource_(), pool_(cuda_resource_, initial_pool_bytes) {}

  raft::device_resources resources_;
  rmm::mr::cuda_memory_resource cuda_resource_;
  rmm::mr::pool_memory_resource pool_;
};

DeviceContext::DeviceContext(int device_id, uint64_t initial_pool_bytes) : device_id_(device_id) {
  DeviceGuard device_guard{device_id_};
  try {
    resources_ = std::make_unique<Resources>(initial_pool_bytes);
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
  auto stream_view = raft::resource::get_cuda_stream(resources_->resources_);
  cudaStream_t stream = stream_view.value();
  auto stream_ref = cuda::stream_ref{stream};
  auto resource = rmm::device_async_resource_ref{resources_->pool_};
  DeviceAllocation device{resource, stream_ref, stream, count * sizeof(uint32_t)};
  brt_launch_smoke(device.data(), count, stream);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    const std::string message = cudaGetErrorString(error);
    (void)device.synchronize();
    try {
      device.deallocate();
    } catch (...) {
    }
    throw std::runtime_error(message);
  }
  std::vector<uint32_t> host(count);
  error = cudaMemcpyAsync(
      host.data(), device.data(), count * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  const cudaError_t synchronization_error = device.synchronize();
  if (error == cudaSuccess) error = synchronization_error;
  if (error != cudaSuccess) {
    const std::string message = cudaGetErrorString(error);
    try {
      device.deallocate();
    } catch (...) {
    }
    throw std::runtime_error(message);
  }
  device.deallocate();
  uint64_t checksum = std::accumulate(host.begin(), host.end(), uint64_t{0});
  BrtSmokeResult result{device_id_, count, checksum};
  device_guard.restore();
  return result;
}

}  // namespace brt
