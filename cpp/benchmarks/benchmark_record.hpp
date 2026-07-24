#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace brt {

struct BenchmarkRecord {
  std::string utc_timestamp;
  std::string project_commit;
  std::string device;
  std::string driver_version;
  std::string cuda_version;
  std::string architecture;
  std::string operator_signature;
  std::string selected_kernel;
  std::string provenance_kind;
  std::optional<std::string> upstream_revision;
  bool correctness_passed{};
  double max_abs_error{};
  double max_rel_error{};
  double cosine_similarity{};
  std::uint64_t nonfinite_mismatches{};
  std::uint64_t warmup_count{};
  std::uint64_t measured_iterations{};
  double median_us{};
  double p95_us{};
  double min_us{};
  double max_us{};
  std::uint64_t workspace_bytes{};
  std::uint64_t launch_count{};
  std::string graph_mode;

  bool performance_publishable() const;
  std::string to_json_line() const;
};

}  // namespace brt
