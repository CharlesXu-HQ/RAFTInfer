#include <brt/c_api.h>

#include <iostream>

int main() {
  BrtEngineConfig config{sizeof(BrtEngineConfig), 0, 64U * 1024U * 1024U};
  BrtEngineHandle* engine = nullptr;
  BrtStatus status = brt_engine_create(&config, &engine);
  if (status.code != BRT_STATUS_OK) {
    std::cerr << status.message << '\n';
    return 1;
  }
  BrtSmokeResult result{};
  status = brt_engine_run_smoke(engine, &result);
  brt_engine_destroy(engine);
  if (status.code != BRT_STATUS_OK) {
    std::cerr << status.message << '\n';
    return 1;
  }
  std::cout << "{\"device_id\":" << result.device_id
            << ",\"element_count\":" << result.element_count
            << ",\"checksum\":" << result.checksum << "}\n";
  return result.checksum == 523776 ? 0 : 2;
}
