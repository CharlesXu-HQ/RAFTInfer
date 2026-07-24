#include "../benchmarks/benchmark_record.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void expect_contains(const std::string& text, const std::string& needle) {
  assert(text.find(needle) != std::string::npos);
}

void expect_not_contains(const std::string& text, const std::string& needle) {
  assert(text.find(needle) == std::string::npos);
}

template <class Fn>
void expect_invalid_argument(Fn&& fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const std::invalid_argument&) {
    thrown = true;
  }
  assert(thrown);
}

brt::BenchmarkRecord complete_record() {
  return brt::BenchmarkRecord{
      .utc_timestamp = "2026-07-25T01:02:03Z",
      .project_commit = "a9283f8",
      .device = "NVIDIA GeForce RTX 5090",
      .driver_version = "580.159.03",
      .cuda_version = "13.2.78",
      .architecture = "sm_120a",
      .operator_signature = "matmul(q4_k[1,4096], bf16[4096,4096]) -> bf16[1,4096]",
      .selected_kernel = "native \"dense\\fast\"\ncontrol\001 utf8中文",
      .provenance_kind = "project_native",
      .upstream_revision = std::nullopt,
      .correctness_passed = true,
      .max_abs_error = 0.001,
      .max_rel_error = 0.002,
      .cosine_similarity = 0.99999,
      .nonfinite_mismatches = 0,
      .warmup_count = 5,
      .measured_iterations = 25,
      .median_us = 17.5,
      .p95_us = 19.25,
      .min_us = 16.75,
      .max_us = 21.0,
      .workspace_bytes = 1048576,
      .launch_count = 3,
      .graph_mode = "disabled"};
}

}  // namespace

int main() {
  const brt::BenchmarkRecord record = complete_record();
  const std::string json = record.to_json_line();

  assert(!json.empty());
  assert(json.back() == '\n');
  assert(std::count(json.begin(), json.end(), '\n') == 1);
  const std::string expected =
      "{\"schema_version\":1,"
      "\"utc_timestamp\":\"2026-07-25T01:02:03Z\","
      "\"project_commit\":\"a9283f8\","
      "\"device\":\"NVIDIA GeForce RTX 5090\","
      "\"driver_version\":\"580.159.03\","
      "\"cuda_version\":\"13.2.78\","
      "\"architecture\":\"sm_120a\","
      "\"operator_signature\":\"matmul(q4_k[1,4096], bf16[4096,4096]) -> bf16[1,4096]\","
      "\"selected_kernel\":\"native \\\"dense\\\\fast\\\"\\ncontrol\\u0001 utf8中文\","
      "\"provenance_kind\":\"project_native\","
      "\"upstream_revision\":null,"
      "\"correctness_passed\":true,"
      "\"max_abs_error\":0.001,"
      "\"max_rel_error\":0.002,"
      "\"cosine_similarity\":0.99999000000000005,"
      "\"nonfinite_mismatches\":0,"
      "\"warmup_count\":5,"
      "\"measured_iterations\":25,"
      "\"median_us\":17.5,"
      "\"p95_us\":19.25,"
      "\"min_us\":16.75,"
      "\"max_us\":21,"
      "\"workspace_bytes\":1048576,"
      "\"launch_count\":3,"
      "\"graph_mode\":\"disabled\","
      "\"performance_publishable\":true}\n";
  assert(json == expected);
  expect_not_contains(json, "dense\\fast");
  expect_not_contains(json, "fast\"\ncontrol");
  assert(record.performance_publishable());

  brt::BenchmarkRecord imported = record;
  imported.provenance_kind = "upstream_bw24";
  imported.upstream_revision = "bw24@abc123";
  expect_contains(imported.to_json_line(), "\"upstream_revision\":\"bw24@abc123\"");

  brt::BenchmarkRecord fractional_timestamp = record;
  fractional_timestamp.utc_timestamp = "2024-02-29T23:59:59.123Z";
  expect_contains(fractional_timestamp.to_json_line(),
                  "\"utc_timestamp\":\"2024-02-29T23:59:59.123Z\"");

  brt::BenchmarkRecord local_offset_timestamp = record;
  local_offset_timestamp.utc_timestamp = "2026-07-25T01:02:03+08:00";
  assert(!local_offset_timestamp.performance_publishable());
  expect_invalid_argument([&] { (void)local_offset_timestamp.to_json_line(); });

  brt::BenchmarkRecord arbitrary_timestamp = record;
  arbitrary_timestamp.utc_timestamp = "yesterday";
  assert(!arbitrary_timestamp.performance_publishable());
  expect_invalid_argument([&] { (void)arbitrary_timestamp.to_json_line(); });

  brt::BenchmarkRecord range_timestamp = record;
  range_timestamp.utc_timestamp = "2026-07-25T24:00:00Z";
  assert(!range_timestamp.performance_publishable());
  expect_invalid_argument([&] { (void)range_timestamp.to_json_line(); });

  brt::BenchmarkRecord invalid_date_timestamp = record;
  invalid_date_timestamp.utc_timestamp = "2023-02-29T01:02:03Z";
  assert(!invalid_date_timestamp.performance_publishable());
  expect_invalid_argument([&] { (void)invalid_date_timestamp.to_json_line(); });

  brt::BenchmarkRecord dot_without_fraction_timestamp = record;
  dot_without_fraction_timestamp.utc_timestamp = "2026-07-25T01:02:03.Z";
  assert(!dot_without_fraction_timestamp.performance_publishable());
  expect_invalid_argument([&] { (void)dot_without_fraction_timestamp.to_json_line(); });

  brt::BenchmarkRecord native_with_revision = record;
  native_with_revision.upstream_revision = "should-not-exist";
  assert(!native_with_revision.performance_publishable());
  expect_invalid_argument([&] { (void)native_with_revision.to_json_line(); });

  brt::BenchmarkRecord imported_without_revision = record;
  imported_without_revision.provenance_kind = "upstream_bw24";
  assert(!imported_without_revision.performance_publishable());
  expect_invalid_argument([&] { (void)imported_without_revision.to_json_line(); });

  brt::BenchmarkRecord imported_empty_revision = record;
  imported_empty_revision.provenance_kind = "upstream_bw24";
  imported_empty_revision.upstream_revision = "";
  assert(!imported_empty_revision.performance_publishable());
  expect_invalid_argument([&] { (void)imported_empty_revision.to_json_line(); });

  brt::BenchmarkRecord failed_correctness = record;
  failed_correctness.correctness_passed = false;
  assert(!failed_correctness.performance_publishable());
  expect_contains(failed_correctness.to_json_line(), "\"performance_publishable\":false");

  brt::BenchmarkRecord no_iterations = record;
  no_iterations.measured_iterations = 0;
  assert(!no_iterations.performance_publishable());

  brt::BenchmarkRecord zero_median = record;
  zero_median.median_us = 0.0;
  assert(!zero_median.performance_publishable());

  brt::BenchmarkRecord negative_p95 = record;
  negative_p95.p95_us = -1.0;
  assert(!negative_p95.performance_publishable());
  expect_invalid_argument([&] { (void)negative_p95.to_json_line(); });

  brt::BenchmarkRecord performance_without_correctness = record;
  performance_without_correctness.correctness_passed = false;
  performance_without_correctness.measured_iterations = 100;
  performance_without_correctness.median_us = 1.0;
  performance_without_correctness.p95_us = 1.0;
  assert(!performance_without_correctness.performance_publishable());

  brt::BenchmarkRecord infinite_metric = record;
  infinite_metric.max_abs_error = std::numeric_limits<double>::infinity();
  expect_invalid_argument([&] { (void)infinite_metric.to_json_line(); });

  brt::BenchmarkRecord negative_metric = record;
  negative_metric.max_rel_error = -0.001;
  assert(!negative_metric.performance_publishable());
  expect_invalid_argument([&] { (void)negative_metric.to_json_line(); });

  brt::BenchmarkRecord invalid_cosine = record;
  invalid_cosine.cosine_similarity = 1.01;
  assert(!invalid_cosine.performance_publishable());
  expect_invalid_argument([&] { (void)invalid_cosine.to_json_line(); });

  brt::BenchmarkRecord nan_timing = record;
  nan_timing.median_us = std::numeric_limits<double>::quiet_NaN();
  assert(!nan_timing.performance_publishable());
  expect_invalid_argument([&] { (void)nan_timing.to_json_line(); });

  brt::BenchmarkRecord negative_timing = record;
  negative_timing.min_us = -0.1;
  assert(!negative_timing.performance_publishable());
  expect_invalid_argument([&] { (void)negative_timing.to_json_line(); });

  brt::BenchmarkRecord unordered_timing = record;
  unordered_timing.p95_us = 16.0;
  assert(!unordered_timing.performance_publishable());
  expect_invalid_argument([&] { (void)unordered_timing.to_json_line(); });

  brt::BenchmarkRecord no_launches = record;
  no_launches.launch_count = 0;
  assert(!no_launches.performance_publishable());
  expect_invalid_argument([&] { (void)no_launches.to_json_line(); });

  brt::BenchmarkRecord missing_identity = record;
  missing_identity.project_commit.clear();
  assert(!missing_identity.performance_publishable());
  expect_invalid_argument([&] { (void)missing_identity.to_json_line(); });
}
