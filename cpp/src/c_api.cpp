#include <brt/c_api.h>

#include "engine.hpp"

#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>

struct BrtEngineHandle { brt::Engine engine; };

namespace {

struct LastError {
  char message[256]{};
};

thread_local LastError g_last_error;

void set_last_error(const char* message) noexcept {
  std::size_t index = 0;
  for (; index + 1 < sizeof(g_last_error.message) && message[index] != '\0'; ++index) {
    g_last_error.message[index] = message[index];
  }
  g_last_error.message[index] = '\0';
}

void clear_last_error() noexcept { g_last_error.message[0] = '\0'; }

}  // namespace

static BrtStatus fail(BrtStatusCode code, const char* message) noexcept {
  set_last_error(message);
  return BrtStatus{code, g_last_error.message};
}

extern "C" BrtStatus brt_engine_create(
    const BrtEngineConfig* config, BrtEngineHandle** out_engine) {
  try {
    clear_last_error();
    if (out_engine == nullptr) {
      return fail(BRT_STATUS_INVALID_ARGUMENT, "config and out_engine are required");
    }
    *out_engine = nullptr;
    if (config == nullptr) {
      return fail(BRT_STATUS_INVALID_ARGUMENT, "config and out_engine are required");
    }
    if (config->struct_size != sizeof(BrtEngineConfig)) {
      return fail(BRT_STATUS_INVALID_ARGUMENT, "BrtEngineConfig size mismatch");
    }
    auto handle = std::make_unique<BrtEngineHandle>(BrtEngineHandle{brt::Engine{*config}});
    *out_engine = handle.release();
    return BrtStatus{BRT_STATUS_OK, nullptr};
  } catch (const std::invalid_argument& error) {
    return fail(BRT_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::exception& error) {
    return fail(BRT_STATUS_INTERNAL, error.what());
  } catch (...) {
    return fail(BRT_STATUS_INTERNAL, "internal error");
  }
}

extern "C" void brt_engine_destroy(BrtEngineHandle* engine) {
  try {
    delete engine;
  } catch (...) {
    fail(BRT_STATUS_INTERNAL, "internal error");
  }
}

extern "C" int32_t brt_engine_is_cuda_enabled(const BrtEngineHandle* engine) {
  try {
    return engine != nullptr && engine->engine.cuda_enabled() ? 1 : 0;
  } catch (...) {
    fail(BRT_STATUS_INTERNAL, "internal error");
    return 0;
  }
}

extern "C" BrtStatus brt_engine_run_smoke(
    BrtEngineHandle* engine, BrtSmokeResult* out_result) {
  clear_last_error();
  if (engine == nullptr || out_result == nullptr) {
    return fail(BRT_STATUS_INVALID_ARGUMENT, "engine and out_result are required");
  }
  if (!engine->engine.cuda_enabled()) {
    return fail(BRT_STATUS_UNAVAILABLE, "CUDA backend is not enabled");
  }
  try {
    *out_result = engine->engine.run_smoke();
    return BrtStatus{BRT_STATUS_OK, nullptr};
  } catch (const std::exception& error) {
    return fail(BRT_STATUS_CUDA_ERROR, error.what());
  } catch (...) {
    return fail(BRT_STATUS_CUDA_ERROR, "CUDA error");
  }
}

extern "C" const char* brt_last_error_message(void) {
  try {
    return g_last_error.message;
  } catch (...) {
    return "internal error";
  }
}
