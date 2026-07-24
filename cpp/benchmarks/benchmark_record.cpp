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
  return correctness_passed && iterations > 0 && std::isfinite(median_us) &&
         std::isfinite(p95_us) && median_us > 0.0 && p95_us > 0.0;
}

std::string BenchmarkRecord::to_json_line() const {
  require_finite(max_absolute_error, "max_absolute_error");
  require_finite(max_relative_error, "max_relative_error");
  require_finite(cosine_similarity, "cosine_similarity");
  require_finite(median_us, "median_us");
  require_finite(p95_us, "p95_us");

  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << '{';
  bool first = true;
  append_number_field(out, "schema_version", 1, first);
  append_string_field(out, "operator_name", operator_name, first);
  append_string_field(out, "kernel_name", kernel_name, first);
  append_string_field(out, "backend", backend, first);
  append_optional_string_field(out, "upstream_revision", upstream_revision, first);
  append_string_field(out, "arch", arch, first);
  append_string_field(out, "dtype", dtype, first);
  append_string_field(out, "shape", shape, first);
  append_bool_field(out, "correctness_passed", correctness_passed, first);
  append_number_field(out, "max_absolute_error", max_absolute_error, first);
  append_number_field(out, "max_relative_error", max_relative_error, first);
  append_number_field(out, "cosine_similarity", cosine_similarity, first);
  append_number_field(out, "nonfinite_mismatches", nonfinite_mismatches, first);
  append_number_field(out, "iterations", iterations, first);
  append_number_field(out, "median_us", median_us, first);
  append_number_field(out, "p95_us", p95_us, first);
  append_bool_field(out, "performance_publishable", performance_publishable(), first);
  out << "}\n";
  return out.str();
}

}  // namespace brt
