#include <brt/c_api.h>

#include <cassert>
#include <cstring>

int main() {
  BrtEngineConfig config{};
  config.struct_size = sizeof(BrtEngineConfig);
  config.device_id = 0;
  config.initial_pool_bytes = 64U * 1024U * 1024U;

  assert(brt_engine_create(nullptr, nullptr).code == BRT_STATUS_INVALID_ARGUMENT);

  BrtEngineHandle* engine = nullptr;
  BrtStatus status = brt_engine_create(&config, &engine);
  assert(status.code == BRT_STATUS_OK);
  assert(engine != nullptr);
  assert(brt_engine_is_cuda_enabled(engine) == 0);

  brt_engine_destroy(engine);
  assert(std::strlen(brt_last_error_message()) == 0);
  return 0;
}
