#include "engine.hpp"

#if BRT_ENABLE_CUDA
#include "../foundation/device_context.hpp"
#endif

#include <stdexcept>

namespace brt {

Engine::Engine(const BrtEngineConfig& config)
    : device_id_(config.device_id), initial_pool_bytes_(config.initial_pool_bytes) {
  if (device_id_ < 0) throw std::invalid_argument("device_id must be non-negative");
  if (initial_pool_bytes_ == 0) throw std::invalid_argument("initial_pool_bytes must be non-zero");
#if BRT_ENABLE_CUDA
  device_ = std::make_unique<DeviceContext>(device_id_, initial_pool_bytes_);
#endif
}

Engine::~Engine() = default;

bool Engine::cuda_enabled() const noexcept {
#if BRT_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

BrtSmokeResult Engine::run_smoke() {
#if BRT_ENABLE_CUDA
  return device_->run_smoke();
#else
  throw std::runtime_error("CUDA backend is not enabled");
#endif
}

}  // namespace brt
