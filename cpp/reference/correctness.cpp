#include "correctness.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brt::reference {

namespace {

bool nonfinite_values_match(float candidate, float reference) {
  return std::isinf(candidate) && std::isinf(reference) &&
         std::signbit(candidate) == std::signbit(reference);
}

}  // namespace

CorrectnessMetrics compare(
    std::span<const float> candidate,
    std::span<const float> reference,
    double relative_floor) {
  if (candidate.size() != reference.size()) {
    throw std::invalid_argument("candidate and reference spans must have equal lengths");
  }
  if (!(relative_floor > 0.0F) || !std::isfinite(relative_floor)) {
    throw std::invalid_argument("relative_floor must be finite and positive");
  }

  CorrectnessMetrics metrics{
      0.0,
      0.0,
      1.0,
      0,
      candidate.size(),
  };

  double dot = 0.0;
  double candidate_norm_sq = 0.0;
  double reference_norm_sq = 0.0;

  for (std::size_t i = 0; i < candidate.size(); ++i) {
    const float candidate_value = candidate[i];
    const float reference_value = reference[i];
    const bool candidate_finite = std::isfinite(candidate_value);
    const bool reference_finite = std::isfinite(reference_value);

    if (!candidate_finite || !reference_finite) {
      if (!nonfinite_values_match(candidate_value, reference_value)) {
        ++metrics.nonfinite_mismatches;
      }
      continue;
    }

    const double candidate_double = static_cast<double>(candidate_value);
    const double reference_double = static_cast<double>(reference_value);
    const double absolute_error = std::fabs(candidate_double - reference_double);
    const double denominator = std::max(std::fabs(reference_double), relative_floor);
    metrics.max_absolute_error = std::max(metrics.max_absolute_error, absolute_error);
    metrics.max_relative_error =
        std::max(metrics.max_relative_error, absolute_error / denominator);

    dot += candidate_double * reference_double;
    candidate_norm_sq += candidate_double * candidate_double;
    reference_norm_sq += reference_double * reference_double;
  }

  if (candidate_norm_sq == 0.0 && reference_norm_sq == 0.0) {
    metrics.cosine_similarity = 1.0;
  } else if (candidate_norm_sq == 0.0 || reference_norm_sq == 0.0) {
    metrics.cosine_similarity = 0.0;
  } else {
    metrics.cosine_similarity =
        dot / (std::sqrt(candidate_norm_sq) * std::sqrt(reference_norm_sq));
    metrics.cosine_similarity = std::clamp(metrics.cosine_similarity, -1.0, 1.0);
  }

  return metrics;
}

bool passes_tolerance(const CorrectnessMetrics& metrics, const Tolerance& tolerance) {
  if (metrics.nonfinite_mismatches != 0) {
    return false;
  }
  if (!std::isfinite(metrics.max_absolute_error) || metrics.max_absolute_error < 0.0 ||
      !std::isfinite(metrics.max_relative_error) || metrics.max_relative_error < 0.0 ||
      !std::isfinite(metrics.cosine_similarity) || metrics.cosine_similarity < -1.0 ||
      metrics.cosine_similarity > 1.0) {
    return false;
  }
  if (tolerance.max_absolute_error &&
      (!std::isfinite(*tolerance.max_absolute_error) || *tolerance.max_absolute_error < 0.0 ||
       metrics.max_absolute_error > *tolerance.max_absolute_error)) {
    return false;
  }
  if (tolerance.max_relative_error &&
      (!std::isfinite(*tolerance.max_relative_error) || *tolerance.max_relative_error < 0.0 ||
       metrics.max_relative_error > *tolerance.max_relative_error)) {
    return false;
  }
  if (tolerance.min_cosine_similarity &&
      (!std::isfinite(*tolerance.min_cosine_similarity) ||
       *tolerance.min_cosine_similarity < -1.0 || *tolerance.min_cosine_similarity > 1.0 ||
       metrics.cosine_similarity < *tolerance.min_cosine_similarity)) {
    return false;
  }
  return true;
}

}  // namespace brt::reference
