#pragma once

#include <brt/c_api.h>

namespace brt {

class Engine {
 public:
  explicit Engine(const BrtEngineConfig& config);
  bool cuda_enabled() const noexcept;

 private:
  int device_id_;
  uint64_t initial_pool_bytes_;
};

}  // namespace brt
