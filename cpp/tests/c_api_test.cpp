#include <brt/c_api.h>

#include <cassert>
#include <cstring>

int main() {
  BrtEngineConfig config{};
  config.struct_size = sizeof(BrtEngineConfig);
  config.device_id = 0;
  config.initial_pool_bytes = 64U * 1024U * 1024U;

  BrtEngineHandle* engine = reinterpret_cast<BrtEngineHandle*>(1);
  BrtStatus status = brt_engine_create(nullptr, &engine);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "config and out_engine are required") == 0);
  assert(engine == nullptr);
  assert(std::strcmp(brt_last_error_message(), "config and out_engine are required") == 0);

  status = brt_engine_create(&config, nullptr);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "config and out_engine are required") == 0);

  BrtEngineConfig invalid_config = config;
  invalid_config.struct_size = 0;
  engine = reinterpret_cast<BrtEngineHandle*>(1);
  status = brt_engine_create(&invalid_config, &engine);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "BrtEngineConfig size mismatch") == 0);
  assert(engine == nullptr);

  invalid_config = config;
  invalid_config.device_id = -1;
  engine = reinterpret_cast<BrtEngineHandle*>(1);
  status = brt_engine_create(&invalid_config, &engine);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "device_id must be non-negative") == 0);
  assert(engine == nullptr);

  invalid_config = config;
  invalid_config.initial_pool_bytes = 0;
  engine = reinterpret_cast<BrtEngineHandle*>(1);
  status = brt_engine_create(&invalid_config, &engine);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "initial_pool_bytes must be non-zero") == 0);
  assert(engine == nullptr);

  engine = nullptr;
  status = brt_engine_create(&config, &engine);
  assert(status.code == BRT_STATUS_OK);
  assert(engine != nullptr);
  assert(brt_engine_is_cuda_enabled(engine) == 0);

  brt_engine_destroy(engine);
  assert(std::strlen(brt_last_error_message()) == 0);
  return 0;
}
