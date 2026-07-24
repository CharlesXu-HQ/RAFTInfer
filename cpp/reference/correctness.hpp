#pragma once

#include <cstddef>
#include <optional>
#include <span>

namespace brt::reference {

struct CorrectnessMetrics {
  float max_absolute_error;
  float max_relative_error;
  float cosine_similarity;
  std::size_t nonfinite_mismatches;
  std::size_t elements;
};

struct Tolerance {
  std::optional<float> max_absolute_error;
  std::optional<float> max_relative_error;
  std::optional<float> min_cosine_similarity;
};

// Relative error uses abs(candidate - reference) / max(abs(reference), relative_floor).
// Cosine similarity treats two finite zero vectors as 1 and exactly one finite
// zero vector as 0. Same-signed infinities are compatible; NaNs and finite /
// non-finite pairs are non-finite mismatches.
CorrectnessMetrics compare(
    std::span<const float> candidate,
    std::span<const float> reference,
    float relative_floor = 1.0e-12F);

// Fails on any non-finite mismatch, then checks only thresholds set by the
// operator-local tolerance.
bool passes_tolerance(const CorrectnessMetrics& metrics, const Tolerance& tolerance);

template <class CandidateIndices, class ReferenceIndices>
bool indices_equal(const CandidateIndices& candidate, const ReferenceIndices& reference) {
  if (candidate.size() != reference.size()) {
    return false;
  }
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    if (candidate[i] != reference[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace brt::reference
