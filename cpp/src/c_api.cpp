#include <brt/c_api.h>

#include "engine.hpp"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

struct BrtEngineHandle { brt::Engine engine; };
static thread_local std::string g_last_error;

static BrtStatus fail(BrtStatusCode code, const char* message) {
  g_last_error = message;
  return BrtStatus{code, g_last_error.c_str()};
}

extern "C" BrtStatus brt_engine_create(
    const BrtEngineConfig* config, BrtEngineHandle** out_engine) {
  g_last_error.clear();
  if (config == nullptr || out_engine == nullptr) {
    return fail(BRT_STATUS_INVALID_ARGUMENT, "config and out_engine are required");
  }
  if (config->struct_size != sizeof(BrtEngineConfig)) {
    return fail(BRT_STATUS_INVALID_ARGUMENT, "BrtEngineConfig size mismatch");
  }
  try {
    auto handle = std::make_unique<BrtEngineHandle>(BrtEngineHandle{brt::Engine{*config}});
    *out_engine = handle.release();
    return BrtStatus{BRT_STATUS_OK, nullptr};
  } catch (const std::invalid_argument& error) {
    return fail(BRT_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::exception& error) {
    return fail(BRT_STATUS_INTERNAL, error.what());
  }
}

extern "C" void brt_engine_destroy(BrtEngineHandle* engine) { delete engine; }

extern "C" int32_t brt_engine_is_cuda_enabled(const BrtEngineHandle* engine) {
  return engine != nullptr && engine->engine.cuda_enabled() ? 1 : 0;
}

extern "C" const char* brt_last_error_message(void) { return g_last_error.c_str(); }
