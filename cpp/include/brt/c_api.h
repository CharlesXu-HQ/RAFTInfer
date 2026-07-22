#pragma once

#include <brt/status.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BrtEngineHandle BrtEngineHandle;

typedef struct BrtEngineConfig {
  size_t struct_size;
  int32_t device_id;
  uint64_t initial_pool_bytes;
} BrtEngineConfig;

BrtStatus brt_engine_create(const BrtEngineConfig* config, BrtEngineHandle** out_engine);
void brt_engine_destroy(BrtEngineHandle* engine);
int32_t brt_engine_is_cuda_enabled(const BrtEngineHandle* engine);
const char* brt_last_error_message(void);

#ifdef __cplusplus
}
#endif
