#pragma once

#include <brt/c_api.h>

#include <cstdint>
#include <memory>

namespace brt::model {
class CudaWeightPlan;
class Model;
struct Qwen35Manifest;
}  // namespace brt::model

namespace brt {

class ExecutionContext;

class DeviceContext {
 public:
  DeviceContext(int device_id, uint64_t initial_pool_bytes);
  ~DeviceContext() noexcept;
  BrtSmokeResult run_smoke();
  std::unique_ptr<model::CudaWeightPlan>
  upload_qwen35_weights(const model::Model& model);
  std::unique_ptr<model::CudaWeightPlan>
  upload_qwen35_weights_for_tests(const model::Model& model,
                                  const model::Qwen35Manifest& manifest);

 private:
  class Resources;

  int device_id_;
  std::shared_ptr<Resources> resources_;
};

}  // namespace brt
