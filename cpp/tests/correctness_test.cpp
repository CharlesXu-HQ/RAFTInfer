#include "../reference/correctness.hpp"

#include "assert_enabled.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

void expect_near(double actual, double expected, double tolerance = 1.0e-6) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > tolerance) {
    assert(false);
  }
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

}  // namespace

int main() {
  using raftinfer::reference::Tolerance;
  using raftinfer::reference::compare;
  using raftinfer::reference::indices_equal;
  using raftinfer::reference::passes_tolerance;

  {
    const std::array<float, 3> values{1.0F, -2.0F, 3.0F};
    const auto metrics = compare(values, values);
    assert(metrics.elements == values.size());
    expect_near(metrics.max_absolute_error, 0.0F);
    expect_near(metrics.max_relative_error, 0.0F);
    expect_near(metrics.cosine_similarity, 1.0F);
    assert(metrics.nonfinite_mismatches == 0);
    assert(passes_tolerance(metrics, Tolerance{.max_absolute_error = 0.0F,
                                               .max_relative_error = 0.0F,
                                               .min_cosine_similarity = 1.0F}));
  }

  {
    const std::array<float, 3> candidate{1.0F, 2.01F, 2.99F};
    const std::array<float, 3> reference{1.0F, 2.0F, 3.0F};
    const auto metrics = compare(candidate, reference);
    expect_near(metrics.max_absolute_error, 0.01F, 1.0e-5F);
    assert(metrics.max_relative_error <= 0.005001F);
    assert(metrics.cosine_similarity > 0.99999F);
    assert(passes_tolerance(metrics, Tolerance{.max_absolute_error = 0.011F,
                                               .max_relative_error = 0.006F,
                                               .min_cosine_similarity = 0.999F}));
    assert(!passes_tolerance(metrics, Tolerance{.max_absolute_error = 0.001F}));
    assert(!passes_tolerance(metrics, Tolerance{.max_relative_error = 0.001F}));
    assert(!passes_tolerance(metrics, Tolerance{.min_cosine_similarity = 0.9999999F}));
  }

  {
    const float infinity = std::numeric_limits<float>::infinity();
    const auto same_infinite = compare(std::array{infinity, -infinity},
                                       std::array{infinity, -infinity});
    assert(same_infinite.nonfinite_mismatches == 0);
    assert(passes_tolerance(same_infinite, Tolerance{}));

    const auto different_infinite = compare(std::array{infinity}, std::array{-infinity});
    assert(different_infinite.nonfinite_mismatches == 1);
    assert(!passes_tolerance(different_infinite, Tolerance{}));

    const auto finite_mismatch = compare(std::array{infinity}, std::array{1.0F});
    assert(finite_mismatch.nonfinite_mismatches == 1);
    assert(!passes_tolerance(finite_mismatch, Tolerance{}));

    const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
    const auto nan_pair = compare(std::array{quiet_nan}, std::array{quiet_nan});
    assert(nan_pair.nonfinite_mismatches == 1);
    assert(!passes_tolerance(nan_pair, Tolerance{}));
  }

  {
    const std::array<float, 3> zeros{0.0F, 0.0F, 0.0F};
    const auto metrics = compare(zeros, zeros);
    expect_near(metrics.cosine_similarity, 1.0F);
    assert(passes_tolerance(metrics, Tolerance{.min_cosine_similarity = 1.0F}));
  }

  {
    const std::array<float, 0> empty{};
    const auto metrics = compare(empty, empty);
    assert(metrics.elements == 0);
    expect_near(metrics.max_absolute_error, 0.0);
    expect_near(metrics.max_relative_error, 0.0);
    expect_near(metrics.cosine_similarity, 1.0);
    assert(metrics.nonfinite_mismatches == 0);
    assert(passes_tolerance(metrics, Tolerance{.min_cosine_similarity = 1.0F}));
  }

  {
    const auto metrics = compare(std::array{0.0F, 0.0F}, std::array{1.0F, 0.0F});
    expect_near(metrics.cosine_similarity, 0.0F);
    assert(!passes_tolerance(metrics, Tolerance{.min_cosine_similarity = 0.1F}));
  }

  {
    expect_invalid_argument([] { (void)compare(std::array{1.0F}, std::array{1.0F, 2.0F}); });
  }

  {
    auto metrics = compare(std::array{1.0e-9F}, std::array{2.0e-9F}, 1.0e-6F);
    expect_near(metrics.max_relative_error, 0.001F);
  }

  {
    const float max = std::numeric_limits<float>::max();
    const auto metrics = compare(std::array{max}, std::array{-max});
    assert(std::isfinite(metrics.max_absolute_error));
    expect_near(metrics.max_relative_error, 2.0);
    assert(metrics.max_absolute_error > 6.0e38);
  }

  {
    const auto exact = compare(std::array{1.0F}, std::array{1.0F});
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    assert(!passes_tolerance(exact, Tolerance{.max_absolute_error = -1.0F}));
    assert(!passes_tolerance(exact, Tolerance{.max_absolute_error = nan}));
    assert(!passes_tolerance(exact, Tolerance{.max_absolute_error = infinity}));

    assert(!passes_tolerance(exact, Tolerance{.max_relative_error = -1.0F}));
    assert(!passes_tolerance(exact, Tolerance{.max_relative_error = nan}));
    assert(!passes_tolerance(exact, Tolerance{.max_relative_error = infinity}));

    assert(!passes_tolerance(exact, Tolerance{.min_cosine_similarity = nan}));
    assert(!passes_tolerance(exact, Tolerance{.min_cosine_similarity = infinity}));
    assert(!passes_tolerance(exact, Tolerance{.min_cosine_similarity = 1.1F}));
    assert(!passes_tolerance(exact, Tolerance{.min_cosine_similarity = -1.1F}));
  }

  {
    assert(indices_equal(std::array<std::int32_t, 4>{0, 2, 2, 7},
                         std::array<std::int32_t, 4>{0, 2, 2, 7}));
    assert(!indices_equal(std::array<std::int32_t, 4>{0, 2, 3, 7},
                          std::array<std::int32_t, 4>{0, 2, 2, 7}));
    assert(!indices_equal(std::array<std::int32_t, 3>{0, 2, 2},
                          std::array<std::int32_t, 4>{0, 2, 2, 7}));
  }
}
