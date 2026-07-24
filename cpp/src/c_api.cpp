#include <brt/c_api.h>

#include "../model/gguf_reader.hpp"
#include "../model/model.hpp"
#include "../model/qwen35_config.hpp"
#include "../model/qwen35_manifest.hpp"
#include "engine.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>

struct BrtEngineHandle {
  explicit BrtEngineHandle(const BrtEngineConfig &config) : engine(config) {}

  brt::Engine engine;
};

struct BrtModelHandle {
  explicit BrtModelHandle(std::unique_ptr<brt::model::Model> value)
      : model(std::move(value)) {}

  std::unique_ptr<brt::model::Model> model;
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

} // namespace

static BrtStatus fail(BrtStatusCode code, const char *message) noexcept {
  set_last_error(message);
  return BrtStatus{code, g_last_error.message};
}

extern "C" BrtStatus brt_engine_create(const BrtEngineConfig *config,
                                       BrtEngineHandle **out_engine) {
  try {
    clear_last_error();
    if (out_engine == nullptr) {
      return fail(BRT_STATUS_INVALID_ARGUMENT,
                  "config and out_engine are required");
    }
    *out_engine = nullptr;
    if (config == nullptr) {
      return fail(BRT_STATUS_INVALID_ARGUMENT,
                  "config and out_engine are required");
    }
    if (config->struct_size != sizeof(BrtEngineConfig)) {
      return fail(BRT_STATUS_INVALID_ARGUMENT, "BrtEngineConfig size mismatch");
    }
    auto handle = std::make_unique<BrtEngineHandle>(*config);
    *out_engine = handle.release();
    return BrtStatus{BRT_STATUS_OK, nullptr};
  } catch (const std::invalid_argument &error) {
    return fail(BRT_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::exception &error) {
    return fail(BRT_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(BRT_STATUS_INTERNAL, "internal error");
  }
}

extern "C" void brt_engine_destroy(BrtEngineHandle *engine) {
  try {
    delete engine;
  } catch (...) {
    fail(BRT_STATUS_INTERNAL, "internal error");
  }
}

extern "C" int32_t brt_engine_is_cuda_enabled(const BrtEngineHandle *engine) {
  try {
    return engine != nullptr && engine->engine.cuda_enabled() ? 1 : 0;
  } catch (...) {
    fail(BRT_STATUS_INTERNAL, "internal error");
    return 0;
  }
}

extern "C" BrtStatus brt_engine_run_smoke(BrtEngineHandle *engine,
                                          BrtSmokeResult *out_result) {
  clear_last_error();
  if (engine == nullptr || out_result == nullptr) {
    return fail(BRT_STATUS_INVALID_ARGUMENT,
                "engine and out_result are required");
  }
  if (!engine->engine.cuda_enabled()) {
    return fail(BRT_STATUS_UNAVAILABLE, "CUDA backend is not enabled");
  }
  try {
    *out_result = engine->engine.run_smoke();
    return BrtStatus{BRT_STATUS_OK, nullptr};
  } catch (const std::exception &error) {
    return fail(BRT_STATUS_CUDA_ERROR, error.what());
  } catch (...) {
    return fail(BRT_STATUS_CUDA_ERROR, "CUDA error");
  }
}

extern "C" BrtStatus brt_engine_load_model(BrtEngineHandle *engine,
                                           const char *gguf_path,
                                           BrtModelHandle **out_model) {
  try {
    clear_last_error();
    if (out_model == nullptr) {
      return fail(BRT_STATUS_INVALID_ARGUMENT, "out_model is required");
    }
    *out_model = nullptr;
    if (engine == nullptr || gguf_path == nullptr || gguf_path[0] == '\0') {
      return fail(BRT_STATUS_INVALID_ARGUMENT,
                  "engine, gguf_path, and out_model are required");
    }
    auto handle =
        std::make_unique<BrtModelHandle>(engine->engine.load_model(gguf_path));
    *out_model = handle.release();
    return BrtStatus{BRT_STATUS_OK, nullptr};
  } catch (const brt::model::ModelIoError &error) {
    return fail(BRT_STATUS_UNAVAILABLE, error.what());
  } catch (const brt::gguf::ParseError &error) {
    return fail(BRT_STATUS_UNSUPPORTED, error.what());
  } catch (const brt::model::ConfigError &error) {
    return fail(BRT_STATUS_UNSUPPORTED, error.what());
  } catch (const brt::model::ManifestError &error) {
    return fail(BRT_STATUS_UNSUPPORTED, error.what());
  } catch (const std::bad_alloc &) {
    return fail(BRT_STATUS_RESOURCE_EXHAUSTED,
                "model loading exhausted host memory");
  } catch (const std::exception &error) {
    return fail(BRT_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(BRT_STATUS_INTERNAL, "internal error");
  }
}

extern "C" void brt_model_destroy(BrtModelHandle *model) {
  try {
    delete model;
  } catch (...) {
    fail(BRT_STATUS_INTERNAL, "internal error");
  }
}

extern "C" BrtStatus brt_model_copy_tokenizer_spec(const BrtModelHandle *model,
                                                   BrtOwnedBuffer *out_buffer) {
  try {
    clear_last_error();
    if (model == nullptr || out_buffer == nullptr) {
      return fail(BRT_STATUS_INVALID_ARGUMENT,
                  "model and out_buffer are required");
    }
    if (out_buffer->struct_size != sizeof(BrtOwnedBuffer)) {
      return fail(BRT_STATUS_INVALID_ARGUMENT, "BrtOwnedBuffer size mismatch");
    }
    if (out_buffer->data != nullptr || out_buffer->size != 0) {
      return fail(BRT_STATUS_INVALID_ARGUMENT, "BrtOwnedBuffer must be empty");
    }

    const auto spec = model->model->tokenizer_spec();
    auto data = std::make_unique<std::uint8_t[]>(spec.size());
    std::copy(spec.begin(), spec.end(), data.get());
    out_buffer->version = 1;
    out_buffer->size = spec.size();
    out_buffer->data = data.release();
    return BrtStatus{BRT_STATUS_OK, nullptr};
  } catch (const std::bad_alloc &) {
    return fail(BRT_STATUS_RESOURCE_EXHAUSTED,
                "tokenizer specification allocation failed");
  } catch (const std::exception &error) {
    return fail(BRT_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(BRT_STATUS_INTERNAL, "internal error");
  }
}

extern "C" void brt_owned_buffer_free(BrtOwnedBuffer *buffer) {
  if (buffer == nullptr) {
    return;
  }
  delete[] buffer->data;
  buffer->version = 0;
  buffer->data = nullptr;
  buffer->size = 0;
}

extern "C" const char *brt_last_error_message(void) {
  try {
    return g_last_error.message;
  } catch (...) {
    return "internal error";
  }
}
