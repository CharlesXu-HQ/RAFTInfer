#include "device_context.hpp"

#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <numeric>
#include <stdexcept>
#include <vector>

extern void brt_launch_smoke(uint32_t* values, uint32_t count, cudaStream_t stream);

namespace brt {

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

BrtSmokeResult DeviceContext::run_smoke() {
  constexpr uint32_t count = 1024;
  auto stream_view = raft::resource::get_cuda_stream(resources_);
  cudaStream_t stream = stream_view.value();
  auto stream_ref = cuda::stream_ref{stream};
  auto resource = rmm::device_async_resource_ref{pool_};
  auto* device = static_cast<uint32_t*>(resource.allocate(stream_ref, count * sizeof(uint32_t)));
  brt_launch_smoke(device, count, stream);
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
  return BrtSmokeResult{device_id_, count, checksum};
}

}  // namespace brt
