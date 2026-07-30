#include <raftinfer/c_api.h>

#include <iostream>

int main() {
  RaftInferEngineConfig config{sizeof(RaftInferEngineConfig), 0, 64U * 1024U * 1024U};
  RaftInferEngineHandle* engine = nullptr;
  RaftInferStatus status = raftinfer_engine_create(&config, &engine);
  if (status.code != RAFTINFER_STATUS_OK) {
    std::cerr << status.message << '\n';
    return 1;
  }
  RaftInferSmokeResult result{};
  status = raftinfer_engine_run_smoke(engine, &result);
  raftinfer_engine_destroy(engine);
  if (status.code != RAFTINFER_STATUS_OK) {
    std::cerr << status.message << '\n';
    return 1;
  }
  std::cout << "{\"device_id\":" << result.device_id
            << ",\"element_count\":" << result.element_count
            << ",\"checksum\":" << result.checksum << "}\n";
  return result.checksum == 523776 ? 0 : 2;
}
