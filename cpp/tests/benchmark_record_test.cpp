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
      .operator_name = "matmul",
      .kernel_name = "native \"dense\\fast\"\ncontrol\001",
      .backend = "native",
      .upstream_revision = std::nullopt,
      .arch = "sm_120a",
      .dtype = "q4_k",
      .shape = "m=1,n=4096,k=4096",
      .correctness_passed = true,
      .max_absolute_error = 0.001,
      .max_relative_error = 0.002,
      .cosine_similarity = 0.99999,
      .nonfinite_mismatches = 0,
      .iterations = 25,
      .median_us = 17.5,
      .p95_us = 19.25};
}

}  // namespace

int main() {
  const brt::BenchmarkRecord record = complete_record();
  const std::string json = record.to_json_line();

  assert(!json.empty());
  assert(json.back() == '\n');
  assert(std::count(json.begin(), json.end(), '\n') == 1);
  expect_contains(json, "{\"schema_version\":1,\"operator_name\":\"matmul\"");
  expect_contains(json, "\"kernel_name\":\"native \\\"dense\\\\fast\\\"\\ncontrol\\u0001\"");
  expect_not_contains(json, "dense\\fast");
  expect_not_contains(json, "fast\"\ncontrol");
  expect_contains(json, "\"upstream_revision\":null");
  expect_contains(json, "\"performance_publishable\":true");
  assert(record.performance_publishable());

  brt::BenchmarkRecord failed_correctness = record;
  failed_correctness.correctness_passed = false;
  assert(!failed_correctness.performance_publishable());
  expect_contains(failed_correctness.to_json_line(), "\"performance_publishable\":false");

  brt::BenchmarkRecord no_iterations = record;
  no_iterations.iterations = 0;
  assert(!no_iterations.performance_publishable());

  brt::BenchmarkRecord zero_median = record;
  zero_median.median_us = 0.0;
  assert(!zero_median.performance_publishable());

  brt::BenchmarkRecord negative_p95 = record;
  negative_p95.p95_us = -1.0;
  assert(!negative_p95.performance_publishable());

  brt::BenchmarkRecord performance_without_correctness = record;
  performance_without_correctness.correctness_passed = false;
  performance_without_correctness.iterations = 100;
  performance_without_correctness.median_us = 1.0;
  performance_without_correctness.p95_us = 1.0;
  assert(!performance_without_correctness.performance_publishable());

  brt::BenchmarkRecord infinite_metric = record;
  infinite_metric.max_absolute_error = std::numeric_limits<double>::infinity();
  expect_invalid_argument([&] { (void)infinite_metric.to_json_line(); });

  brt::BenchmarkRecord nan_timing = record;
  nan_timing.median_us = std::numeric_limits<double>::quiet_NaN();
  assert(!nan_timing.performance_publishable());
  expect_invalid_argument([&] { (void)nan_timing.to_json_line(); });
}
