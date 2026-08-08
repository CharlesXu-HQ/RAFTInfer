#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <functional>

namespace raftinfer {

class CudaGraphDecode {
public:
  using CaptureBody = std::function<void()>;

  CudaGraphDecode(int device_id, cudaStream_t stream);
  ~CudaGraphDecode() noexcept;

  void capture(CaptureBody body);
  void replay();
  void replay_on_current_device();
  void reset() noexcept;
  bool captured() const noexcept;

  CudaGraphDecode(const CudaGraphDecode &) = delete;
  CudaGraphDecode &operator=(const CudaGraphDecode &) = delete;

private:
  int device_id_{};
  cudaStream_t stream_{};
  cudaGraph_t graph_{};
  cudaGraphExec_t exec_{};
};

namespace test {

void reset_cuda_graph_decode_construction_count() noexcept;
std::size_t cuda_graph_decode_construction_count() noexcept;

} // namespace test

} // namespace raftinfer
