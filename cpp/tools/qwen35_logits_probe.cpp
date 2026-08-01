#include "../execution/qwen35_executor.hpp"
#include "../foundation/device_context.hpp"
#include "../model/model.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::size_t parse_size(std::string_view value, const char *name) {
  std::size_t result = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() ||
      result == 0) {
    throw std::invalid_argument(std::string{name} +
                                " must be a positive integer");
  }
  return result;
}

std::int32_t parse_token(std::string_view value) {
  std::int32_t result = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() ||
      result < 0) {
    throw std::invalid_argument("token IDs must be non-negative integers");
  }
  return result;
}

void print_json_float(float value) {
  if (std::isfinite(value)) {
    std::cout << value;
  } else {
    std::cout << "null";
  }
}

void print_top_logits(std::span<const float> logits, std::size_t top_k) {
  std::vector<std::size_t> indices(logits.size());
  for (std::size_t index = 0; index < indices.size(); ++index) {
    indices[index] = index;
  }
  const auto better = [&](std::size_t left, std::size_t right) {
    const bool left_is_finite = std::isfinite(logits[left]);
    const bool right_is_finite = std::isfinite(logits[right]);
    if (left_is_finite != right_is_finite) {
      return left_is_finite;
    }
    if (!left_is_finite) {
      return left < right;
    }
    if (logits[left] != logits[right]) {
      return logits[left] > logits[right];
    }
    return left < right;
  };
  top_k = std::min(top_k, indices.size());
  std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                    better);
  std::cout << "\"top_logits\":[";
  for (std::size_t rank = 0; rank < top_k; ++rank) {
    if (rank != 0) {
      std::cout << ',';
    }
    const auto index = indices[rank];
    std::cout << "{\"token\":" << index << ",\"logit\":";
    print_json_float(logits[index]);
    std::cout << ",\"finite\":"
              << (std::isfinite(logits[index]) ? "true" : "false") << '}';
  }
  std::cout << ']';
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 5) {
      std::cerr << "usage: raftinfer-qwen35-logits MODEL CONTEXT TOP_K TOKEN...\n";
      return 2;
    }
    const std::string model_path{argv[1]};
    const std::size_t context = parse_size(argv[2], "context");
    const std::size_t top_k = parse_size(argv[3], "top_k");
    std::vector<std::int32_t> tokens;
    tokens.reserve(static_cast<std::size_t>(argc - 4));
    for (int index = 4; index < argc; ++index) {
      tokens.push_back(parse_token(argv[index]));
    }
    if (tokens.size() > context) {
      throw std::invalid_argument("token count exceeds context");
    }

    raftinfer::model::Model model{model_path};
    raftinfer::DeviceContext device{0, 256U * 1024U * 1024U};
    auto weights = device.upload_qwen35_weights(model);
    auto owner = device.create_execution_owner(
        raftinfer::Qwen35Executor::workspace_bytes(model.qwen35_config(), context));
    auto execution_context = owner->execution_context();
    raftinfer::Qwen35Executor executor{execution_context, model.qwen35_config(),
                                 *weights, context};
    const bool trace_enabled = std::getenv("RAFTINFER_QWEN35_TRACE") != nullptr;
    executor.enable_trace(trace_enabled);

    const auto prefill = executor.prefill(tokens);
    std::vector<float> prefill_logits(model.qwen35_config().vocabulary_size);
    executor.copy_last_logits(prefill_logits);
    if (trace_enabled) {
      for (const auto &entry : executor.trace()) {
        std::cerr << "TRACE " << entry.name << " sum=" << entry.sum
                  << " first=" << entry.first[0] << ',' << entry.first[1]
                  << ',' << entry.first[2] << " last=" << entry.last[0] << ','
                  << entry.last[1] << ',' << entry.last[2] << '\n';
      }
      executor.enable_trace(false);
    }

    executor.reset();
    raftinfer::Qwen35ExecutorResult stepped{};
    for (const auto token : tokens) {
      stepped = executor.decode(token);
    }
    std::vector<float> stepped_logits(model.qwen35_config().vocabulary_size);
    executor.copy_last_logits(stepped_logits);

    float max_absolute_difference = 0.0F;
    std::size_t max_difference_token = 0;
    bool non_finite_difference = false;
    for (std::size_t index = 0; index < prefill_logits.size(); ++index) {
      if (!std::isfinite(prefill_logits[index]) ||
          !std::isfinite(stepped_logits[index])) {
        if (!non_finite_difference) {
          max_difference_token = index;
        }
        non_finite_difference = true;
        continue;
      }
      const float difference =
          std::fabs(prefill_logits[index] - stepped_logits[index]);
      if (difference > max_absolute_difference) {
        max_absolute_difference = difference;
        max_difference_token = index;
      }
    }

    std::cout << "{\"schema_version\":2,\"prefill_token\":" << prefill.token
              << ",\"stepped_token\":" << stepped.token
              << ",\"max_absolute_difference\":";
    if (non_finite_difference) {
      std::cout << "null";
    } else {
      print_json_float(max_absolute_difference);
    }
    std::cout << ",\"non_finite_difference\":"
              << (non_finite_difference ? "true" : "false")
              << ",\"max_difference_token\":" << max_difference_token << ',';
    print_top_logits(prefill_logits, top_k);
    std::cout << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "raftinfer-qwen35-logits: " << error.what() << '\n';
    return 1;
  }
}
