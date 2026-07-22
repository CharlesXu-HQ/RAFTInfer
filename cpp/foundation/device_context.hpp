#pragma once

#include <brt/c_api.h>
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cstdint>

namespace brt {

class DeviceContext {
 public:
  DeviceContext(int device_id, uint64_t initial_pool_bytes);
  BrtSmokeResult run_smoke();

 private:
  static int select_device(int device_id);
  int device_id_;
  raft::device_resources resources_;
  rmm::mr::cuda_memory_resource cuda_resource_;
  rmm::mr::pool_memory_resource pool_;
};

}  // namespace brt
