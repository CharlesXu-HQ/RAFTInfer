#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace raftinfer::model {
class Model;
}

namespace raftinfer::reference {

struct Qwen35ReferenceExecution {
  std::vector<float> logits;
  std::int32_t token{};
};

Qwen35ReferenceExecution
qwen35_execute_model(const model::Model &model,
                     std::span<const std::int32_t> tokens);

} // namespace raftinfer::reference
