#include "engine.hpp"

#include <stdexcept>

namespace brt {

Engine::Engine(const BrtEngineConfig& config)
    : device_id_(config.device_id), initial_pool_bytes_(config.initial_pool_bytes) {
  if (device_id_ < 0) throw std::invalid_argument("device_id must be non-negative");
  if (initial_pool_bytes_ == 0) throw std::invalid_argument("initial_pool_bytes must be non-zero");
}

bool Engine::cuda_enabled() const noexcept {
#if BRT_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

}  // namespace brt
