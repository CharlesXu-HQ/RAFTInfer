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

typedef struct BrtSmokeResult {
  int32_t device_id;
  uint32_t element_count;
  uint64_t checksum;
} BrtSmokeResult;

BrtStatus brt_engine_create(const BrtEngineConfig* config, BrtEngineHandle** out_engine);
void brt_engine_destroy(BrtEngineHandle* engine);
int32_t brt_engine_is_cuda_enabled(const BrtEngineHandle* engine);
BrtStatus brt_engine_run_smoke(BrtEngineHandle* engine, BrtSmokeResult* out_result);
const char* brt_last_error_message(void);

#ifdef __cplusplus
}
#endif
