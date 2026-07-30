#pragma once

#include <raftinfer/c_api.h>
#include <memory>
#include <string>

namespace raftinfer {

namespace model {
class Model;
}

#if RAFTINFER_ENABLE_CUDA
class DeviceContext;
#endif

class Engine {
public:
  explicit Engine(const RaftInferEngineConfig &config);
  ~Engine();
  bool cuda_enabled() const noexcept;
  RaftInferSmokeResult run_smoke();
  std::uint64_t peak_allocated_gpu_bytes();
  std::shared_ptr<model::Model> load_model(const std::string &gguf_path) const;

private:
  int device_id_;
  uint64_t initial_pool_bytes_;
#if RAFTINFER_ENABLE_CUDA
  std::shared_ptr<DeviceContext> device_;
#endif
};

} // namespace raftinfer
