#include "benchmark_record.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace brt {
namespace {

void require_finite(double value, std::string_view field) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(field) + " must be finite");
  }
}

void require_nonempty(std::string_view value, std::string_view field) {
  if (value.empty()) {
    throw std::invalid_argument(std::string(field) + " must be nonempty");
  }
}

void require_nonnegative(double value, std::string_view field) {
  require_finite(value, field);
  if (value < 0.0) {
    throw std::invalid_argument(std::string(field) + " must be nonnegative");
  }
}

void require_positive(double value, std::string_view field) {
  require_finite(value, field);
  if (value <= 0.0) {
    throw std::invalid_argument(std::string(field) + " must be positive");
  }
}

void validate_record(const BenchmarkRecord& record) {
  require_nonempty(record.utc_timestamp, "utc_timestamp");
  require_nonempty(record.project_commit, "project_commit");
  require_nonempty(record.device, "device");
  require_nonempty(record.driver_version, "driver_version");
  require_nonempty(record.cuda_version, "cuda_version");
  require_nonempty(record.architecture, "architecture");
  require_nonempty(record.operator_signature, "operator_signature");
  require_nonempty(record.selected_kernel, "selected_kernel");
  require_nonempty(record.provenance_kind, "provenance_kind");
  require_nonempty(record.graph_mode, "graph_mode");

  require_nonnegative(record.max_abs_error, "max_abs_error");
  require_nonnegative(record.max_rel_error, "max_rel_error");
  require_finite(record.cosine_similarity, "cosine_similarity");
  if (record.cosine_similarity < -1.0 || record.cosine_similarity > 1.0) {
    throw std::invalid_argument("cosine_similarity must be in [-1, 1]");
  }

  require_finite(record.median_us, "median_us");
  require_finite(record.p95_us, "p95_us");
  require_nonnegative(record.min_us, "min_us");
  require_nonnegative(record.max_us, "max_us");

  if (record.measured_iterations > 0) {
    require_positive(record.min_us, "min_us");
    require_positive(record.median_us, "median_us");
    require_positive(record.p95_us, "p95_us");
    require_positive(record.max_us, "max_us");
    if (record.launch_count == 0) {
      throw std::invalid_argument("launch_count must be positive when measured_iterations > 0");
    }
  }

  if (record.min_us > record.median_us || record.median_us > record.p95_us ||
      record.p95_us > record.max_us) {
    throw std::invalid_argument("latency fields must satisfy min_us <= median_us <= p95_us <= max_us");
  }
}

void append_hex4(std::ostringstream& out, unsigned char value) {
  constexpr char digits[] = "0123456789abcdef";
  out << "\\u00" << digits[(value >> 4U) & 0x0FU] << digits[value & 0x0FU];
}

std::string json_string(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20U) {
          append_hex4(out, ch);
        } else {
          out << static_cast<char>(ch);
        }
        break;
    }
  }
  out << '"';
  return out.str();
}

void append_field_prefix(std::ostringstream& out, std::string_view name, bool first) {
  if (!first) {
    out << ',';
  }
  out << '"' << name << "\":";
}

void append_string_field(
    std::ostringstream& out, std::string_view name, std::string_view value, bool& first) {
  append_field_prefix(out, name, first);
  out << json_string(value);
  first = false;
}

void append_optional_string_field(
    std::ostringstream& out,
    std::string_view name,
    const std::optional<std::string>& value,
    bool& first) {
  append_field_prefix(out, name, first);
  if (value.has_value()) {
    out << json_string(*value);
  } else {
    out << "null";
  }
  first = false;
}

void append_bool_field(std::ostringstream& out, std::string_view name, bool value, bool& first) {
  append_field_prefix(out, name, first);
  out << (value ? "true" : "false");
  first = false;
}

template <class Value>
void append_number_field(
    std::ostringstream& out, std::string_view name, Value value, bool& first) {
  append_field_prefix(out, name, first);
  out << value;
  first = false;
}

}  // namespace

bool BenchmarkRecord::performance_publishable() const {
  try {
    validate_record(*this);
  } catch (const std::invalid_argument&) {
    return false;
  }
  return correctness_passed && measured_iterations > 0 && median_us > 0.0 && p95_us > 0.0;
}

std::string BenchmarkRecord::to_json_line() const {
  validate_record(*this);

  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << '{';
  bool first = true;
  append_number_field(out, "schema_version", 1, first);
  append_string_field(out, "utc_timestamp", utc_timestamp, first);
  append_string_field(out, "project_commit", project_commit, first);
  append_string_field(out, "device", device, first);
  append_string_field(out, "driver_version", driver_version, first);
  append_string_field(out, "cuda_version", cuda_version, first);
  append_string_field(out, "architecture", architecture, first);
  append_string_field(out, "operator_signature", operator_signature, first);
  append_string_field(out, "selected_kernel", selected_kernel, first);
  append_string_field(out, "provenance_kind", provenance_kind, first);
  append_optional_string_field(out, "upstream_revision", upstream_revision, first);
  append_bool_field(out, "correctness_passed", correctness_passed, first);
  append_number_field(out, "max_abs_error", max_abs_error, first);
  append_number_field(out, "max_rel_error", max_rel_error, first);
  append_number_field(out, "cosine_similarity", cosine_similarity, first);
  append_number_field(out, "nonfinite_mismatches", nonfinite_mismatches, first);
  append_number_field(out, "warmup_count", warmup_count, first);
  append_number_field(out, "measured_iterations", measured_iterations, first);
  append_number_field(out, "median_us", median_us, first);
  append_number_field(out, "p95_us", p95_us, first);
  append_number_field(out, "min_us", min_us, first);
  append_number_field(out, "max_us", max_us, first);
  append_number_field(out, "workspace_bytes", workspace_bytes, first);
  append_number_field(out, "launch_count", launch_count, first);
  append_string_field(out, "graph_mode", graph_mode, first);
  append_bool_field(out, "performance_publishable", performance_publishable(), first);
  out << "}\n";
  return out.str();
}

}  // namespace brt
