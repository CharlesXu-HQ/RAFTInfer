#pragma once

#include <raftinfer/status.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RaftInferEngineHandle RaftInferEngineHandle;
typedef struct RaftInferModelHandle RaftInferModelHandle;
typedef struct RaftInferSessionHandle RaftInferSessionHandle;

typedef struct RaftInferEngineConfig {
  size_t struct_size;
  int32_t device_id;
  uint64_t initial_pool_bytes;
} RaftInferEngineConfig;

typedef struct RaftInferSmokeResult {
  int32_t device_id;
  uint32_t element_count;
  uint64_t checksum;
} RaftInferSmokeResult;

typedef struct RaftInferOwnedBuffer {
  size_t struct_size;
  /* Tokenizer-spec wire-format version. Version 1 starts with "RIFTOK\0\1". */
  uint32_t version;
  uint8_t *data;
  size_t size;
} RaftInferOwnedBuffer;

enum {
  RAFTINFER_QWEN35_ATTENTION_MATERIALIZED_REFERENCE = 0,
  RAFTINFER_QWEN35_ATTENTION_ONLINE_TILED = 1,
};

enum {
  RAFTINFER_QWEN35_KV_CACHE_F32 = 0,
  RAFTINFER_QWEN35_KV_CACHE_BF16 = 1,
};

enum {
  RAFTINFER_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR = 0,
  RAFTINFER_QWEN35_KV_CACHE_LAYOUT_HEAD_MAJOR = 1,
};

enum {
  RAFTINFER_QWEN35_DECODE_ATTENTION_SINGLE_BLOCK = 0,
  RAFTINFER_QWEN35_DECODE_ATTENTION_SPLIT_K = 1,
};

typedef struct RaftInferQwen35ExecutionPolicy {
  size_t struct_size;
  uint32_t attention;
  uint32_t kv_cache_dtype;
  uint32_t kv_cache_layout;
  int32_t decode_graph;
  int32_t grouped_input_casts;
} RaftInferQwen35ExecutionPolicy;

typedef struct RaftInferSessionConfig {
  size_t struct_size;
  uint32_t max_context_tokens;
  const RaftInferQwen35ExecutionPolicy *qwen35_policy;
} RaftInferSessionConfig;

typedef struct RaftInferSessionDiagnostics {
  size_t struct_size;
  uint32_t attention;
  uint32_t kv_cache_dtype;
  uint32_t kv_cache_layout;
  int32_t decode_graph_enabled;
  int32_t decode_graph_captured;
  int32_t decode_graph_replayed;
  size_t attention_workspace_bytes;
  uint32_t decode_attention;
  size_t decode_attention_partition_tokens;
  size_t decode_attention_threshold_tokens;
  size_t decode_attention_context_bucket_tokens;
  int32_t decode_attention_split_k_graph_captured;
} RaftInferSessionDiagnostics;

typedef struct RaftInferTokenResult {
  int32_t token_id;
  uint32_t position;
} RaftInferTokenResult;

RaftInferStatus raftinfer_engine_create(const RaftInferEngineConfig *config,
                            RaftInferEngineHandle **out_engine);
void raftinfer_engine_destroy(RaftInferEngineHandle *engine);
int32_t raftinfer_engine_is_cuda_enabled(const RaftInferEngineHandle *engine);
RaftInferStatus raftinfer_engine_run_smoke(RaftInferEngineHandle *engine,
                               RaftInferSmokeResult *out_result);
RaftInferStatus raftinfer_engine_peak_allocated_gpu_bytes(
    RaftInferEngineHandle *engine, uint64_t *out_peak_allocated_gpu_bytes);
RaftInferStatus raftinfer_engine_load_model(RaftInferEngineHandle *engine, const char *gguf_path,
                                RaftInferModelHandle **out_model);
void raftinfer_model_destroy(RaftInferModelHandle *model);
RaftInferStatus raftinfer_model_copy_tokenizer_spec(const RaftInferModelHandle *model,
                                        RaftInferOwnedBuffer *out_buffer);
RaftInferStatus raftinfer_session_create(RaftInferModelHandle *model,
                             const RaftInferSessionConfig *config,
                             RaftInferSessionHandle **out_session);
RaftInferStatus raftinfer_session_prefill(RaftInferSessionHandle *session, const int32_t *tokens,
                              size_t token_count,
                              RaftInferTokenResult *out_result);
RaftInferStatus raftinfer_session_decode(RaftInferSessionHandle *session, int32_t token_id,
                             RaftInferTokenResult *out_result);
RaftInferStatus raftinfer_session_decode_greedy(RaftInferSessionHandle *session,
                                    int32_t first_token_id,
                                    int32_t *out_token_ids,
                                    size_t token_count,
                                    RaftInferTokenResult *out_result);
RaftInferStatus raftinfer_session_diagnostics(RaftInferSessionHandle *session,
                                  RaftInferSessionDiagnostics *out_diagnostics);
RaftInferStatus raftinfer_session_reset(RaftInferSessionHandle *session);
void raftinfer_session_destroy(RaftInferSessionHandle *session);
void raftinfer_owned_buffer_free(RaftInferOwnedBuffer *buffer);
const char *raftinfer_last_error_message(void);

#ifdef __cplusplus
}
#endif
