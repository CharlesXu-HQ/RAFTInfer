#include "cuda_graph_decode.hpp"

#include <stdexcept>
#include <string>

namespace raftinfer {
namespace {

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{"CUDA graph decode "} + operation +
                             ": " + cudaGetErrorString(status));
  }
}

void launch_graph(cudaGraphExec_t exec, cudaStream_t stream) {
  check_cuda(cudaGraphLaunch(exec, stream), "launch failed");
}

} // namespace

CudaGraphDecode::CudaGraphDecode(int device_id, cudaStream_t stream)
    : device_id_(device_id), stream_(stream) {
  if (stream_ == nullptr)
    throw std::runtime_error("CUDA graph decode stream is null");
}

CudaGraphDecode::~CudaGraphDecode() noexcept { reset(); }

void CudaGraphDecode::capture(CaptureBody body) {
  if (!body)
    throw std::runtime_error("CUDA graph decode capture body is empty");
  reset();
  check_cuda(cudaSetDevice(device_id_), "device selection failed");
  check_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
             "capture begin failed");
  try {
    body();
  } catch (...) {
    cudaGraph_t abandoned{};
    if (cudaStreamEndCapture(stream_, &abandoned) == cudaSuccess &&
        abandoned != nullptr) {
      (void)cudaGraphDestroy(abandoned);
    }
    throw;
  }

  cudaGraph_t graph{};
  const cudaError_t end_status = cudaStreamEndCapture(stream_, &graph);
  if (end_status != cudaSuccess) {
    if (graph != nullptr)
      (void)cudaGraphDestroy(graph);
    check_cuda(end_status, "capture end failed");
  }
  cudaGraphExec_t exec{};
  const cudaError_t instantiate_status =
      cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  if (instantiate_status != cudaSuccess) {
    (void)cudaGraphDestroy(graph);
    check_cuda(instantiate_status, "instantiate failed");
  }
  graph_ = graph;
  exec_ = exec;
}

void CudaGraphDecode::replay() {
  if (!captured())
    throw std::runtime_error("CUDA graph decode has not been captured");
  check_cuda(cudaSetDevice(device_id_), "device selection failed");
  launch_graph(exec_, stream_);
}

void CudaGraphDecode::replay_on_current_device() {
  if (!captured())
    throw std::runtime_error("CUDA graph decode has not been captured");
  launch_graph(exec_, stream_);
}

void CudaGraphDecode::reset() noexcept {
  (void)cudaSetDevice(device_id_);
  if (exec_ != nullptr) {
    (void)cudaGraphExecDestroy(exec_);
    exec_ = nullptr;
  }
  if (graph_ != nullptr) {
    (void)cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
}

bool CudaGraphDecode::captured() const noexcept {
  return graph_ != nullptr && exec_ != nullptr;
}

} // namespace raftinfer
