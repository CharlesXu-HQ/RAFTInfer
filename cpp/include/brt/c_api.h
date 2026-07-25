#pragma once

#include <brt/status.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BrtEngineHandle BrtEngineHandle;
typedef struct BrtModelHandle BrtModelHandle;
typedef struct BrtSessionHandle BrtSessionHandle;

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

typedef struct BrtOwnedBuffer {
  size_t struct_size;
  /* Tokenizer-spec wire-format version. Version 1 starts with "BRTTOK\0\1". */
  uint32_t version;
  uint8_t *data;
  size_t size;
} BrtOwnedBuffer;

typedef struct BrtSessionConfig {
  size_t struct_size;
  uint32_t max_context_tokens;
} BrtSessionConfig;

typedef struct BrtTokenResult {
  int32_t token_id;
  uint32_t position;
} BrtTokenResult;

BrtStatus brt_engine_create(const BrtEngineConfig *config,
                            BrtEngineHandle **out_engine);
void brt_engine_destroy(BrtEngineHandle *engine);
int32_t brt_engine_is_cuda_enabled(const BrtEngineHandle *engine);
BrtStatus brt_engine_run_smoke(BrtEngineHandle *engine,
                               BrtSmokeResult *out_result);
BrtStatus brt_engine_load_model(BrtEngineHandle *engine, const char *gguf_path,
                                BrtModelHandle **out_model);
void brt_model_destroy(BrtModelHandle *model);
BrtStatus brt_model_copy_tokenizer_spec(const BrtModelHandle *model,
                                        BrtOwnedBuffer *out_buffer);
BrtStatus brt_session_create(BrtModelHandle *model,
                             const BrtSessionConfig *config,
                             BrtSessionHandle **out_session);
BrtStatus brt_session_prefill(BrtSessionHandle *session, const int32_t *tokens,
                              size_t token_count,
                              BrtTokenResult *out_result);
BrtStatus brt_session_decode(BrtSessionHandle *session, int32_t token_id,
                             BrtTokenResult *out_result);
BrtStatus brt_session_reset(BrtSessionHandle *session);
void brt_session_destroy(BrtSessionHandle *session);
void brt_owned_buffer_free(BrtOwnedBuffer *buffer);
const char *brt_last_error_message(void);

#ifdef __cplusplus
}
#endif
