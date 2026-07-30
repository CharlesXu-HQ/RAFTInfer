#include "engine.hpp"

#include "../model/model.hpp"

#if RAFTINFER_ENABLE_CUDA
#include "../foundation/device_context.hpp"
#include "../model/cuda_weights.hpp"
#endif

#include <stdexcept>

namespace raftinfer {

Engine::Engine(const RaftInferEngineConfig &config)
    : device_id_(config.device_id),
      initial_pool_bytes_(config.initial_pool_bytes) {
  if (device_id_ < 0)
    throw std::invalid_argument("device_id must be non-negative");
  if (initial_pool_bytes_ == 0)
    throw std::invalid_argument("initial_pool_bytes must be non-zero");
#if RAFTINFER_ENABLE_CUDA
  device_ = std::make_shared<DeviceContext>(device_id_, initial_pool_bytes_);
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

std::uint64_t Engine::peak_allocated_gpu_bytes() {
#if RAFTINFER_ENABLE_CUDA
  return device_->peak_allocated_bytes();
#else
  throw std::runtime_error("CUDA backend is not enabled");
#endif
}

std::shared_ptr<model::Model>
Engine::load_model(const std::string &gguf_path) const {
  auto model = std::make_shared<model::Model>(gguf_path);
#if RAFTINFER_ENABLE_CUDA
  auto cuda_weights = device_->upload_qwen35_weights(*model);
  model->attach_cuda(
      device_,
      std::shared_ptr<model::CudaWeightPlan>(std::move(cuda_weights)));
#endif
  return model;
}

} // namespace raftinfer
