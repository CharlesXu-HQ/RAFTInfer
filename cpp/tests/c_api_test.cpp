#include <brt/c_api.h>

#include "assert_enabled.hpp"
#include "qwen35_gguf_fixture.hpp"

#include <cassert>
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
  invalid_session_config.max_context_tokens = 0;
  session = reinterpret_cast<BrtSessionHandle *>(1);
  status = brt_session_create(model, &invalid_session_config, &session);
  assert(status.code == BRT_STATUS_INVALID_ARGUMENT);
  assert(session == nullptr);

  session = nullptr;
  status = brt_session_create(model, &session_config, &session);
  assert(status.code == BRT_STATUS_OK);
  assert(session != nullptr);

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

  status = brt_session_decode(session, token, &result);
  assert(status.code == BRT_STATUS_UNAVAILABLE);
  assert(result.token_id == 123);
  assert(result.position == 456);

  const std::int32_t tokens[] = {1, 2, 3};
  status = brt_session_prefill(session, tokens, 3, &result);
  assert(status.code == BRT_STATUS_UNAVAILABLE);
  assert(result.token_id == 123);
  assert(result.position == 456);

  status = brt_session_reset(session);
  assert(status.code == BRT_STATUS_OK);

  brt_session_destroy(session);

  brt_engine_destroy(engine);
  assert(std::strlen(brt_last_error_message()) == 0);
  return 0;
}
