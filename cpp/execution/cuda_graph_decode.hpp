#pragma once

#include <cuda_runtime_api.h>

#include <functional>

namespace brt {

class CudaGraphDecode {
public:
  using CaptureBody = std::function<void()>;

  CudaGraphDecode(int device_id, cudaStream_t stream);
  ~CudaGraphDecode() noexcept;

  void capture(CaptureBody body);
  void replay();
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

} // namespace brt
