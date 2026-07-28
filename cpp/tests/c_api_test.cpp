#include <brt/c_api.h>

#include "assert_enabled.hpp"
#include "qwen35_gguf_fixture.hpp"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

class TemporaryFile {
public:
  explicit TemporaryFile(const std::vector<std::uint8_t> &bytes) {
    auto pattern =
        (std::filesystem::temp_directory_path() / "brt-qwen35-XXXXXX").string();
    const int descriptor = mkstemp(pattern.data());
    assert(descriptor >= 0);
    close(descriptor);
    path_ = std::move(pattern);
    std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    assert(stream.good());
  }

  ~TemporaryFile() { std::filesystem::remove(path_); }

  const std::string &path() const { return path_; }

private:
  std::string path_;
};

} // namespace

int main() {
#if BRT_TEST_CUDA_ENABLED
  const char *opt_in = std::getenv("BRT_RUN_GPU_TESTS");
  if (opt_in == nullptr || std::strcmp(opt_in, "1") != 0) {
    return 77;
  }
#endif
  BrtEngineConfig config{};
  config.struct_size = sizeof(BrtEngineConfig);
  config.device_id = 0;
  config.initial_pool_bytes = 64U * 1024U * 1024U;

  BrtEngineHandle *engine = reinterpret_cast<BrtEngineHandle *>(1);
  BrtStatus status = brt_engine_create(nullptr, &engine);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "config and out_engine are required") ==
         0);
  assert(engine == nullptr);
  assert(std::strcmp(brt_last_error_message(),
                     "config and out_engine are required") == 0);

  status = brt_engine_create(&config, nullptr);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "config and out_engine are required") ==
         0);

  BrtEngineConfig invalid_config = config;
  invalid_config.struct_size = 0;
  engine = reinterpret_cast<BrtEngineHandle *>(1);
  status = brt_engine_create(&invalid_config, &engine);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "BrtEngineConfig size mismatch") == 0);
  assert(engine == nullptr);

  invalid_config = config;
  invalid_config.device_id = -1;
  engine = reinterpret_cast<BrtEngineHandle *>(1);
  status = brt_engine_create(&invalid_config, &engine);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "device_id must be non-negative") == 0);
  assert(engine == nullptr);

  invalid_config = config;
  invalid_config.initial_pool_bytes = 0;
  engine = reinterpret_cast<BrtEngineHandle *>(1);
  status = brt_engine_create(&invalid_config, &engine);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(status.message, "initial_pool_bytes must be non-zero") ==
         0);
  assert(engine == nullptr);

  engine = nullptr;
  status = brt_engine_create(&config, &engine);
  assert(status.code == BRT_STATUS_OK);
  assert(engine != nullptr);
  assert(brt_engine_is_cuda_enabled(engine) == BRT_TEST_CUDA_ENABLED);

  std::uint64_t peak_allocated_bytes = 123;
  status = brt_engine_peak_allocated_gpu_bytes(nullptr, &peak_allocated_bytes);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(peak_allocated_bytes == 123);

  status = brt_engine_peak_allocated_gpu_bytes(engine, nullptr);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);

  status =
      brt_engine_peak_allocated_gpu_bytes(engine, &peak_allocated_bytes);
#if BRT_TEST_CUDA_ENABLED
  assert(status.code == BRT_STATUS_OK);
  assert(peak_allocated_bytes >= 1024U * 1024U);
  const std::uint64_t engine_peak_allocated_bytes = peak_allocated_bytes;
#else
  assert(status.code == BRT_STATUS_UNAVAILABLE);
  assert(peak_allocated_bytes == 123);
#endif

  BrtModelHandle *model = reinterpret_cast<BrtModelHandle *>(1);
  status = brt_engine_load_model(engine, nullptr, &model);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(model == nullptr);

  status = brt_engine_load_model(engine, "/missing/brt-qwen35.gguf", &model);
  assert(status.code != BRT_STATUS_OK);
  assert(model == nullptr);

  const TemporaryFile fixture(brt::test::make_qwen35_gguf_fixture());
  status = brt_engine_load_model(engine, fixture.path().c_str(), &model);
  assert(status.code == BRT_STATUS_OK);
  assert(model != nullptr);
#if BRT_TEST_CUDA_ENABLED
  status =
      brt_engine_peak_allocated_gpu_bytes(engine, &peak_allocated_bytes);
  assert(status.code == BRT_STATUS_OK);
  assert(peak_allocated_bytes > engine_peak_allocated_bytes);
  const std::uint64_t model_peak_allocated_bytes = peak_allocated_bytes;
#endif

  BrtOwnedBuffer buffer{};
  buffer.struct_size = sizeof(BrtOwnedBuffer);
  status = brt_model_copy_tokenizer_spec(model, &buffer);
  assert(status.code == BRT_STATUS_OK);
  assert(buffer.version == 1);
  assert(buffer.data != nullptr);
  assert(buffer.size > 8);
  assert(std::memcmp(buffer.data, "BRTTOK", 6) == 0);

  brt_owned_buffer_free(&buffer);
  assert(buffer.data == nullptr);
  assert(buffer.size == 0);
  assert(buffer.version == 0);

  BrtSessionHandle *session = reinterpret_cast<BrtSessionHandle *>(1);
  BrtSessionConfig session_config{};
  session_config.struct_size = sizeof(BrtSessionConfig);
  session_config.max_context_tokens = 8;

  status = brt_session_create(nullptr, &session_config, &session);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(session == nullptr);

  session = reinterpret_cast<BrtSessionHandle *>(1);
  status = brt_session_create(model, nullptr, &session);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(session == nullptr);

  BrtSessionConfig invalid_session_config = session_config;
  invalid_session_config.struct_size = 0;
  session = reinterpret_cast<BrtSessionHandle *>(1);
  status = brt_session_create(model, &invalid_session_config, &session);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(session == nullptr);

  invalid_session_config = session_config;
  invalid_session_config.struct_size =
      offsetof(BrtSessionConfig, qwen35_policy) - 1;
  session = reinterpret_cast<BrtSessionHandle *>(1);
  status = brt_session_create(model, &invalid_session_config, &session);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(session == nullptr);

  invalid_session_config = session_config;
  invalid_session_config.max_context_tokens = 0;
  session = reinterpret_cast<BrtSessionHandle *>(1);
  status = brt_session_create(model, &invalid_session_config, &session);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(session == nullptr);

  BrtQwen35ExecutionPolicy invalid_policy{};
  invalid_policy.struct_size = 0;
  BrtSessionConfig policy_config = session_config;
  policy_config.qwen35_policy = &invalid_policy;
  session = reinterpret_cast<BrtSessionHandle *>(1);
  status = brt_session_create(model, &policy_config, &session);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(session == nullptr);

  invalid_policy.struct_size = sizeof(BrtQwen35ExecutionPolicy);
  invalid_policy.attention = BRT_QWEN35_ATTENTION_ONLINE_TILED;
  invalid_policy.kv_cache_dtype = 999;
  invalid_policy.kv_cache_layout = BRT_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR;
  invalid_policy.decode_graph = 1;
  invalid_policy.grouped_input_casts = 1;
  session = reinterpret_cast<BrtSessionHandle *>(1);
  status = brt_session_create(model, &policy_config, &session);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(session == nullptr);

  invalid_policy.kv_cache_dtype = BRT_QWEN35_KV_CACHE_BF16;
  invalid_policy.kv_cache_layout = 999;
  session = reinterpret_cast<BrtSessionHandle *>(1);
  status = brt_session_create(model, &policy_config, &session);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(session == nullptr);

  session = nullptr;
  session_config.struct_size = offsetof(BrtSessionConfig, qwen35_policy);
  status = brt_session_create(model, &session_config, &session);
  assert(status.code == BRT_STATUS_OK);
  assert(session != nullptr);
#if BRT_TEST_CUDA_ENABLED
  status =
      brt_engine_peak_allocated_gpu_bytes(engine, &peak_allocated_bytes);
  assert(status.code == BRT_STATUS_OK);
  assert(peak_allocated_bytes > model_peak_allocated_bytes);
  const std::uint64_t session_peak_allocated_bytes = peak_allocated_bytes;
#endif

  BrtQwen35ExecutionPolicy explicit_policy{};
  explicit_policy.struct_size = sizeof(BrtQwen35ExecutionPolicy);
  explicit_policy.attention = BRT_QWEN35_ATTENTION_ONLINE_TILED;
  explicit_policy.kv_cache_dtype = BRT_QWEN35_KV_CACHE_BF16;
  explicit_policy.kv_cache_layout = BRT_QWEN35_KV_CACHE_LAYOUT_HEAD_MAJOR;
  explicit_policy.decode_graph = 1;
  explicit_policy.grouped_input_casts = 1;
  BrtSessionConfig explicit_session_config{};
  explicit_session_config.struct_size = sizeof(BrtSessionConfig);
  explicit_session_config.max_context_tokens = 8;
  explicit_session_config.qwen35_policy = &explicit_policy;
  BrtSessionHandle *explicit_session = nullptr;
  status = brt_session_create(model, &explicit_session_config, &explicit_session);
  assert(status.code == BRT_STATUS_OK);
  assert(explicit_session != nullptr);

  brt_model_destroy(model);
  model = nullptr;

  BrtTokenResult result{.token_id = 123, .position = 456};
  status = brt_session_decode(nullptr, 7, &result);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(result.token_id == 123);
  assert(result.position == 456);

  status = brt_session_decode(session, 7, nullptr);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);

  status = brt_session_prefill(session, nullptr, 1, &result);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(result.token_id == 123);
  assert(result.position == 456);

  const std::int32_t token = 3;
  status = brt_session_prefill(session, &token, 0, &result);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(result.token_id == 123);
  assert(result.position == 456);

  status = brt_session_decode(session, -1, &result);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(result.token_id == 123);
  assert(result.position == 456);

  status = brt_session_decode(session, 16, &result);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(result.token_id == 123);
  assert(result.position == 456);

  const std::int32_t tokens[] = {1, 2, 3};
  BrtSessionDiagnostics diagnostics{};
  diagnostics.struct_size = sizeof(BrtSessionDiagnostics);
  status = brt_session_diagnostics(nullptr, &diagnostics);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  status = brt_session_diagnostics(session, nullptr);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
#if BRT_TEST_CUDA_ENABLED
  // The tiny ABI fixture is intentionally unsupported by the release online
  // kernel shape; release-shape executor tests cover policy preservation.
  status = brt_session_diagnostics(session, &diagnostics);
  assert(status.code == BRT_STATUS_OK);
  assert(diagnostics.attention ==
         BRT_QWEN35_ATTENTION_MATERIALIZED_REFERENCE);
  assert(diagnostics.kv_cache_dtype == BRT_QWEN35_KV_CACHE_F32);
  assert(diagnostics.kv_cache_layout == BRT_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR);
  assert(diagnostics.decode_graph_enabled == 0);
  assert(diagnostics.decode_graph_captured == 0);
  assert(diagnostics.decode_graph_replayed == 0);
  assert(diagnostics.attention_workspace_bytes > 0);

  BrtSessionDiagnostics explicit_diagnostics{};
  explicit_diagnostics.struct_size = sizeof(BrtSessionDiagnostics);
  status = brt_session_diagnostics(explicit_session, &explicit_diagnostics);
  assert(status.code == BRT_STATUS_OK);
  assert(explicit_diagnostics.attention ==
         BRT_QWEN35_ATTENTION_MATERIALIZED_REFERENCE);
  assert(explicit_diagnostics.kv_cache_dtype == BRT_QWEN35_KV_CACHE_F32);
  assert(explicit_diagnostics.kv_cache_layout ==
         BRT_QWEN35_KV_CACHE_LAYOUT_TOKEN_MAJOR);
  assert(explicit_diagnostics.decode_graph_enabled == 0);
  assert(explicit_diagnostics.decode_graph_captured == 0);
  assert(explicit_diagnostics.decode_graph_replayed == 0);
  assert(explicit_diagnostics.attention_workspace_bytes > 0);

  status = brt_session_decode(session, token, &result);
  assert(status.code == BRT_STATUS_OK);
  assert(result.token_id == 0);
  assert(result.position == 0);

  status = brt_session_prefill(session, tokens, 3, &result);
  assert(status.code == BRT_STATUS_OK);
  assert(result.token_id == 0);
  assert(result.position == 3);

  status = brt_session_reset(session);
  assert(status.code == BRT_STATUS_OK);

  status = brt_session_prefill(session, tokens, 3, &result);
  assert(status.code == BRT_STATUS_OK);
  assert(result.token_id == 0);
  assert(result.position == 2);

  status =
      brt_engine_peak_allocated_gpu_bytes(engine, &peak_allocated_bytes);
  assert(status.code == BRT_STATUS_OK);
  assert(peak_allocated_bytes == session_peak_allocated_bytes);
#else
  status = brt_session_diagnostics(session, &diagnostics);
  assert(status.code == BRT_STATUS_UNAVAILABLE);

  status = brt_session_decode(session, token, &result);
  assert(status.code == BRT_STATUS_UNAVAILABLE);
  assert(result.token_id == 123);
  assert(result.position == 456);

  status = brt_session_prefill(session, tokens, 3, &result);
  assert(status.code == BRT_STATUS_UNAVAILABLE);
  assert(result.token_id == 123);
  assert(result.position == 456);
#endif

  status = brt_session_reset(session);
  assert(status.code == BRT_STATUS_OK);

  brt_session_destroy(session);
  brt_session_destroy(explicit_session);

  brt_engine_destroy(engine);
  assert(std::strlen(brt_last_error_message()) == 0);
  return 0;
}
