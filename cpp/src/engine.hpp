#pragma once

#include <brt/c_api.h>
#include <memory>

namespace brt {

#if BRT_ENABLE_CUDA
class DeviceContext;
#endif

class Engine {
 public:
  explicit Engine(const BrtEngineConfig& config);
  ~Engine();
  bool cuda_enabled() const noexcept;
  BrtSmokeResult run_smoke();

 private:
  int device_id_;
  uint64_t initial_pool_bytes_;
#if BRT_ENABLE_CUDA
  std::unique_ptr<DeviceContext> device_;
#endif
};

}  // namespace brt
