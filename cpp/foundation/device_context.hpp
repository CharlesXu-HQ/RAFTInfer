#pragma once

#include <brt/c_api.h>

#include <cstdint>
#include <memory>

namespace brt {

class DeviceContext {
 public:
  DeviceContext(int device_id, uint64_t initial_pool_bytes);
  ~DeviceContext() noexcept;
  BrtSmokeResult run_smoke();

 private:
  class Resources;

  int device_id_;
  std::unique_ptr<Resources> resources_;
};

}  // namespace brt
