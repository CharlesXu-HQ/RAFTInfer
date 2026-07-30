#include <raftinfer/c_api.h>

#include "../model/gguf_reader.hpp"
#include "../model/model.hpp"
#include "../model/qwen35_config.hpp"
#include "../model/qwen35_manifest.hpp"
#include "../execution/session.hpp"
#include "engine.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>

struct RaftInferEngineHandle {
  explicit RaftInferEngineHandle(const RaftInferEngineConfig &config) : engine(config) {}

  raftinfer::Engine engine;
};

struct RaftInferModelHandle {
  explicit RaftInferModelHandle(std::shared_ptr<raftinfer::model::Model> value)
      : model(std::move(value)) {}

  std::shared_ptr<raftinfer::model::Model> model;
};

struct RaftInferSessionHandle {
  explicit RaftInferSessionHandle(std::unique_ptr<raftinfer::Session> value)
      : session(std::move(value)) {}

  std::unique_ptr<raftinfer::Session> session;
};

namespace {

struct LastError {
  char message[256]{};
};

thread_local LastError g_last_error;

void set_last_error(const char *message) noexcept {
  std::size_t index = 0;
  for (; index + 1 < sizeof(g_last_error.message) && message[index] != '\0';
       ++index) {
    g_last_error.message[index] = message[index];
  }
  g_last_error.message[index] = '\0';
}

void clear_last_error() noexcept { g_last_error.message[0] = '\0'; }

constexpr std::size_t kLegacySessionConfigSize =
    offsetof(RaftInferSessionConfig, qwen35_policy);
constexpr std::size_t kSessionConfigPolicyPointerEnd =
    offsetof(RaftInferSessionConfig, qwen35_policy) +
    sizeof(const RaftInferQwen35ExecutionPolicy *);

raftinfer::Qwen35AttentionImplementation
to_attention_implementation(std::uint32_t value) {
  switch (value) {
  case RAFTINFER_QWEN35_ATTENTION_MATERIALIZED_REFERENCE:
    return raftinfer::Qwen35AttentionImplementation::materialized_reference;
  case RAFTINFER_QWEN35_ATTENTION_ONLINE_TILED:
    return raftinfer::Qwen35AttentionImplementation::online_tiled;
  default:
    throw std::invalid_argument("unknown Qwen3.5 attention implementation");
  }
}

raftinfer::Qwen35KvCacheDType to_kv_cache_dtype(std::uint32_t value) {
  switch (value) {
  case RAFTINFER_QWEN35_KV_CACHE_F32:
    return raftinfer::Qwen35KvCacheDType::f32;
  case RAFTINFER_QWEN35_KV_CACHE_BF16:
    return raftinfer::Qwen35KvCacheDType::bf16;
  default:
    throw std::invalid_argument("unknown Qwen3.5 KV cache dtype");
  }
}

raftinfer::Qwen35KvCacheLayout to_kv_cache_layout(std::uint32_t value) {
  switch (value) {
  case RAFTINFER_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR:
    return raftinfer::Qwen35KvCacheLayout::token_major;
  case RAFTINFER_QWEN35_KV_CACHE_LAYOUT_HEAD_MAJOR:
    return raftinfer::Qwen35KvCacheLayout::head_major;
  default:
    throw std::invalid_argument("unknown Qwen3.5 KV cache layout");
  }
}

std::uint32_t
from_attention_implementation(raftinfer::Qwen35AttentionImplementation value) {
  switch (value) {
  case raftinfer::Qwen35AttentionImplementation::materialized_reference:
    return RAFTINFER_QWEN35_ATTENTION_MATERIALIZED_REFERENCE;
  case raftinfer::Qwen35AttentionImplementation::online_tiled:
    return RAFTINFER_QWEN35_ATTENTION_ONLINE_TILED;
  }
  throw std::logic_error("unmapped Qwen3.5 attention implementation");
}

std::uint32_t from_kv_cache_dtype(raftinfer::Qwen35KvCacheDType value) {
  switch (value) {
  case raftinfer::Qwen35KvCacheDType::f32:
    return RAFTINFER_QWEN35_KV_CACHE_F32;
  case raftinfer::Qwen35KvCacheDType::bf16:
    return RAFTINFER_QWEN35_KV_CACHE_BF16;
  }
  throw std::logic_error("unmapped Qwen3.5 KV cache dtype");
}

std::uint32_t from_kv_cache_layout(raftinfer::Qwen35KvCacheLayout value) {
  switch (value) {
  case raftinfer::Qwen35KvCacheLayout::token_major:
    return RAFTINFER_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR;
  case raftinfer::Qwen35KvCacheLayout::head_major:
    return RAFTINFER_QWEN35_KV_CACHE_LAYOUT_HEAD_MAJOR;
  }
  throw std::logic_error("unmapped Qwen3.5 KV cache layout");
}

bool to_bool(int32_t value, const char *field_name) {
  if (value == 0) {
    return false;
  }
  if (value == 1) {
    return true;
  }
  throw std::invalid_argument(field_name);
}

raftinfer::Qwen35ExecutionPolicy
to_execution_policy(const RaftInferQwen35ExecutionPolicy *policy) {
  raftinfer::Qwen35ExecutionPolicy result{};
  if (policy == nullptr) {
    return result;
  }
  if (policy->struct_size != sizeof(RaftInferQwen35ExecutionPolicy)) {
    throw std::invalid_argument("RaftInferQwen35ExecutionPolicy size mismatch");
  }
  result.attention = to_attention_implementation(policy->attention);
  result.kv_cache = to_kv_cache_dtype(policy->kv_cache_dtype);
  result.kv_cache_layout = to_kv_cache_layout(policy->kv_cache_layout);
  result.decode_graph =
      to_bool(policy->decode_graph, "decode_graph must be 0 or 1");
  result.grouped_input_casts = to_bool(
      policy->grouped_input_casts, "grouped_input_casts must be 0 or 1");
  return result;
}

} // namespace

static RaftInferStatus fail(RaftInferStatusCode code, const char *message) noexcept {
  set_last_error(message);
  return RaftInferStatus{code, g_last_error.message};
}

extern "C" RaftInferStatus raftinfer_engine_create(const RaftInferEngineConfig *config,
                                       RaftInferEngineHandle **out_engine) {
  try {
    clear_last_error();
    if (out_engine == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "config and out_engine are required");
    }
    *out_engine = nullptr;
    if (config == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "config and out_engine are required");
    }
    if (config->struct_size != sizeof(RaftInferEngineConfig)) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "RaftInferEngineConfig size mismatch");
    }
    auto handle = std::make_unique<RaftInferEngineHandle>(*config);
    *out_engine = handle.release();
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::invalid_argument &error) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" void raftinfer_engine_destroy(RaftInferEngineHandle *engine) {
  try {
    delete engine;
  } catch (...) {
    fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" int32_t raftinfer_engine_is_cuda_enabled(const RaftInferEngineHandle *engine) {
  try {
    return engine != nullptr && engine->engine.cuda_enabled() ? 1 : 0;
  } catch (...) {
    fail(RAFTINFER_STATUS_INTERNAL, "internal error");
    return 0;
  }
}

extern "C" RaftInferStatus raftinfer_engine_run_smoke(RaftInferEngineHandle *engine,
                                          RaftInferSmokeResult *out_result) {
  clear_last_error();
  if (engine == nullptr || out_result == nullptr) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                "engine and out_result are required");
  }
  if (!engine->engine.cuda_enabled()) {
    return fail(RAFTINFER_STATUS_UNAVAILABLE, "CUDA backend is not enabled");
  }
  try {
    *out_result = engine->engine.run_smoke();
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, "CUDA error");
  }
}

extern "C" RaftInferStatus raftinfer_engine_peak_allocated_gpu_bytes(
    RaftInferEngineHandle *engine, uint64_t *out_peak_allocated_gpu_bytes) {
  clear_last_error();
  if (engine == nullptr || out_peak_allocated_gpu_bytes == nullptr) {
    return fail(
        RAFTINFER_STATUS_INVALID_ARGUMENT,
        "engine and out_peak_allocated_gpu_bytes are required");
  }
  if (!engine->engine.cuda_enabled()) {
    return fail(RAFTINFER_STATUS_UNAVAILABLE, "CUDA backend is not enabled");
  }
  try {
    *out_peak_allocated_gpu_bytes =
        engine->engine.peak_allocated_gpu_bytes();
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, "CUDA error");
  }
}

extern "C" RaftInferStatus raftinfer_engine_load_model(RaftInferEngineHandle *engine,
                                           const char *gguf_path,
                                           RaftInferModelHandle **out_model) {
  try {
    clear_last_error();
    if (out_model == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "out_model is required");
    }
    *out_model = nullptr;
    if (engine == nullptr || gguf_path == nullptr || gguf_path[0] == '\0') {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "engine, gguf_path, and out_model are required");
    }
    auto handle =
        std::make_unique<RaftInferModelHandle>(engine->engine.load_model(gguf_path));
    *out_model = handle.release();
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const raftinfer::model::ModelIoError &error) {
    return fail(RAFTINFER_STATUS_UNAVAILABLE, error.what());
  } catch (const raftinfer::gguf::ParseError &error) {
    return fail(RAFTINFER_STATUS_UNSUPPORTED, error.what());
  } catch (const raftinfer::model::ConfigError &error) {
    return fail(RAFTINFER_STATUS_UNSUPPORTED, error.what());
  } catch (const raftinfer::model::ManifestError &error) {
    return fail(RAFTINFER_STATUS_UNSUPPORTED, error.what());
  } catch (const std::bad_alloc &) {
    return fail(RAFTINFER_STATUS_RESOURCE_EXHAUSTED,
                "model loading exhausted host memory");
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" void raftinfer_model_destroy(RaftInferModelHandle *model) {
  try {
    delete model;
  } catch (...) {
    fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" RaftInferStatus raftinfer_model_copy_tokenizer_spec(const RaftInferModelHandle *model,
                                                   RaftInferOwnedBuffer *out_buffer) {
  try {
    clear_last_error();
    if (model == nullptr || out_buffer == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "model and out_buffer are required");
    }
    if (out_buffer->struct_size != sizeof(RaftInferOwnedBuffer)) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "RaftInferOwnedBuffer size mismatch");
    }
    if (out_buffer->data != nullptr || out_buffer->size != 0) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "RaftInferOwnedBuffer must be empty");
    }

    const auto spec = model->model->tokenizer_spec();
    auto data = std::make_unique<std::uint8_t[]>(spec.size());
    std::copy(spec.begin(), spec.end(), data.get());
    out_buffer->version = 1;
    out_buffer->size = spec.size();
    out_buffer->data = data.release();
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::bad_alloc &) {
    return fail(RAFTINFER_STATUS_RESOURCE_EXHAUSTED,
                "tokenizer specification allocation failed");
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" RaftInferStatus raftinfer_session_create(RaftInferModelHandle *model,
                                         const RaftInferSessionConfig *config,
                                         RaftInferSessionHandle **out_session) {
  try {
    clear_last_error();
    if (out_session == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "out_session is required");
    }
    *out_session = nullptr;
    if (model == nullptr || config == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "model, config, and out_session are required");
    }
    if (config->struct_size < kLegacySessionConfigSize ||
        (config->struct_size > kLegacySessionConfigSize &&
         config->struct_size < kSessionConfigPolicyPointerEnd)) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "RaftInferSessionConfig size mismatch");
    }
    if (config->max_context_tokens == 0) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "max_context_tokens must be non-zero");
    }
    if (model->model == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "model is invalid");
    }
    const RaftInferQwen35ExecutionPolicy *policy_config = nullptr;
    if (config->struct_size >= kSessionConfigPolicyPointerEnd) {
      policy_config = config->qwen35_policy;
    }
    const auto policy = to_execution_policy(policy_config);
    auto session = std::make_unique<RaftInferSessionHandle>(
        std::make_unique<raftinfer::Session>(model->model,
                                       config->max_context_tokens, policy));
    *out_session = session.release();
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::bad_alloc &) {
    return fail(RAFTINFER_STATUS_RESOURCE_EXHAUSTED,
                "session allocation exhausted host memory");
  } catch (const std::invalid_argument &error) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::length_error &error) {
    return fail(RAFTINFER_STATUS_RESOURCE_EXHAUSTED, error.what());
  } catch (const raftinfer::SessionCudaError &error) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, error.what());
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" RaftInferStatus raftinfer_session_prefill(RaftInferSessionHandle *session,
                                          const int32_t *tokens,
                                          size_t token_count,
                                          RaftInferTokenResult *out_result) {
  try {
    clear_last_error();
    if (session == nullptr || tokens == nullptr || out_result == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "session, tokens, and out_result are required");
    }
    if (token_count == 0) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "prefill token_count must be non-zero");
    }
    const auto result =
        session->session->prefill({tokens, token_count});
    out_result->token_id = result.token_id;
    out_result->position = result.position;
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::invalid_argument &error) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const raftinfer::SessionUnavailableError &error) {
    return fail(RAFTINFER_STATUS_UNAVAILABLE, error.what());
  } catch (const raftinfer::SessionCudaError &error) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, error.what());
  } catch (const std::bad_alloc &) {
    return fail(RAFTINFER_STATUS_RESOURCE_EXHAUSTED,
                "prefill exhausted host memory");
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" RaftInferStatus raftinfer_session_decode(RaftInferSessionHandle *session,
                                         int32_t token_id,
                                         RaftInferTokenResult *out_result) {
  try {
    clear_last_error();
    if (session == nullptr || out_result == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "session and out_result are required");
    }
    const auto result = session->session->decode(token_id);
    out_result->token_id = result.token_id;
    out_result->position = result.position;
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::invalid_argument &error) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const raftinfer::SessionUnavailableError &error) {
    return fail(RAFTINFER_STATUS_UNAVAILABLE, error.what());
  } catch (const raftinfer::SessionCudaError &error) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, error.what());
  } catch (const std::bad_alloc &) {
    return fail(RAFTINFER_STATUS_RESOURCE_EXHAUSTED,
                "decode exhausted host memory");
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" RaftInferStatus
raftinfer_session_decode_greedy(RaftInferSessionHandle *session, int32_t first_token_id,
                          int32_t *out_token_ids, size_t token_count,
                          RaftInferTokenResult *out_result) {
  try {
    clear_last_error();
    if (session == nullptr || out_token_ids == nullptr ||
        out_result == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "session, out_token_ids, and out_result are required");
    }
    if (token_count == 0) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "decode token_count must be non-zero");
    }
    const auto result =
        session->session->decode_greedy(first_token_id,
                                        {out_token_ids, token_count});
    out_result->token_id = result.token_id;
    out_result->position = result.position;
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const std::invalid_argument &error) {
    return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const raftinfer::SessionUnavailableError &error) {
    return fail(RAFTINFER_STATUS_UNAVAILABLE, error.what());
  } catch (const raftinfer::SessionCudaError &error) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, error.what());
  } catch (const std::bad_alloc &) {
    return fail(RAFTINFER_STATUS_RESOURCE_EXHAUSTED,
                "decode exhausted host memory");
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" RaftInferStatus
raftinfer_session_diagnostics(RaftInferSessionHandle *session,
                        RaftInferSessionDiagnostics *out_diagnostics) {
  try {
    clear_last_error();
    if (session == nullptr || out_diagnostics == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "session and out_diagnostics are required");
    }
    if (out_diagnostics->struct_size != sizeof(RaftInferSessionDiagnostics)) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT,
                  "RaftInferSessionDiagnostics size mismatch");
    }
    const auto diagnostics = session->session->diagnostics();
    out_diagnostics->attention =
        from_attention_implementation(diagnostics.attention);
    out_diagnostics->kv_cache_dtype =
        from_kv_cache_dtype(diagnostics.kv_cache);
    out_diagnostics->kv_cache_layout =
        from_kv_cache_layout(diagnostics.kv_cache_layout);
    out_diagnostics->decode_graph_enabled =
        diagnostics.decode_graph_enabled ? 1 : 0;
    out_diagnostics->decode_graph_captured =
        diagnostics.decode_graph_captured ? 1 : 0;
    out_diagnostics->decode_graph_replayed =
        diagnostics.decode_graph_replayed ? 1 : 0;
    out_diagnostics->attention_workspace_bytes =
        diagnostics.attention_workspace_bytes;
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const raftinfer::SessionUnavailableError &error) {
    return fail(RAFTINFER_STATUS_UNAVAILABLE, error.what());
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" RaftInferStatus raftinfer_session_reset(RaftInferSessionHandle *session) {
  try {
    clear_last_error();
    if (session == nullptr) {
      return fail(RAFTINFER_STATUS_INVALID_ARGUMENT, "session is required");
    }
    session->session->reset();
    return RaftInferStatus{RAFTINFER_STATUS_OK, nullptr};
  } catch (const raftinfer::SessionCudaError &error) {
    return fail(RAFTINFER_STATUS_CUDA_ERROR, error.what());
  } catch (const std::exception &error) {
    return fail(RAFTINFER_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" void raftinfer_session_destroy(RaftInferSessionHandle *session) {
  try {
    delete session;
  } catch (...) {
    fail(RAFTINFER_STATUS_INTERNAL, "internal error");
  }
}

extern "C" void raftinfer_owned_buffer_free(RaftInferOwnedBuffer *buffer) {
  if (buffer == nullptr) {
    return;
  }
  delete[] buffer->data;
  buffer->version = 0;
  buffer->data = nullptr;
  buffer->size = 0;
}

extern "C" const char *raftinfer_last_error_message(void) {
  try {
    return g_last_error.message;
  } catch (...) {
    return "internal error";
  }
}
