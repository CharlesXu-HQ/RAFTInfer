#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace brt {

struct BenchmarkRecord {
  std::string operator_name;
  std::string kernel_name;
  std::string backend;
  std::optional<std::string> upstream_revision;
  std::string arch;
  std::string dtype;
  std::string shape;
  bool correctness_passed{};
  double max_absolute_error{};
  double max_relative_error{};
  double cosine_similarity{};
  std::uint64_t nonfinite_mismatches{};
  std::uint64_t iterations{};
  double median_us{};
  double p95_us{};

  bool performance_publishable() const;
  std::string to_json_line() const;
};

}  // namespace brt
