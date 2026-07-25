#pragma once

#include <brt/c_api.h>
#include <memory>
#include <string>

namespace brt {

namespace model {
class Model;
}

#if BRT_ENABLE_CUDA
class DeviceContext;
#endif

class Engine {
public:
  explicit Engine(const BrtEngineConfig &config);
  ~Engine();
  bool cuda_enabled() const noexcept;
  BrtSmokeResult run_smoke();
  std::shared_ptr<model::Model> load_model(const std::string &gguf_path) const;

private:
  int device_id_;
  uint64_t initial_pool_bytes_;
#if BRT_ENABLE_CUDA
  std::shared_ptr<DeviceContext> device_;
#endif
};

} // namespace brt
