#include "../reference/bf16.hpp"
#include "../reference/operators.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

void expect_near(float actual, float expected, float tolerance = 1.0e-5F) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "expected " << expected << " but got " << actual
              << " with tolerance " << tolerance << "\n";
    assert(false);
  }
}

void expect_vector_near(
    std::span<const float> actual,
    std::span<const float> expected,
    float tolerance = 1.0e-5F) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    expect_near(actual[i], expected[i], tolerance);
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

float silu(float x) {
  return x / (1.0F + std::exp(-x));
}

}  // namespace

int main() {
  using brt::reference::AddShape;
  using brt::reference::EmbeddingShape;
  using brt::reference::LinearShape;
  using brt::reference::RmsNormShape;
  using brt::reference::RopeShape;
  using brt::reference::SoftmaxShape;
  using brt::reference::SwiGluShape;

  {
    const std::array<float, 4> lhs{1.0F, -2.0F, 0.5F, 4.0F};
    const std::array<float, 4> rhs{3.0F, 2.0F, -0.5F, -1.0F};
    std::array<float, 4> out{};
    brt::reference::add(lhs, rhs, out, AddShape{4});
    expect_vector_near(out, std::array<float, 4>{4.0F, 0.0F, 0.0F, 3.0F});
  }

  {
    const std::array<std::int32_t, 3> tokens{2, 0, 1};
    const std::array<float, 12> table{
        0.0F, 0.1F, 0.2F, 0.3F,
        1.0F, 1.1F, 1.2F, 1.3F,
        2.0F, 2.1F, 2.2F, 2.3F};
    std::array<float, 12> out{};
    brt::reference::embedding(tokens, table, out, EmbeddingShape{3, 4, 3});
    expect_vector_near(out, std::array<float, 12>{
                               2.0F, 2.1F, 2.2F, 2.3F,
                               0.0F, 0.1F, 0.2F, 0.3F,
                               1.0F, 1.1F, 1.2F, 1.3F});
  }

  {
    const std::array<float, 5> logits{0.5F, 4.0F, 4.0F, -3.0F, 2.0F};
    assert(brt::reference::argmax(logits) == 1);
  }

  {
    const std::array<float, 4> x{3.0F, 4.0F, 1.0F, 2.0F};
    const std::array<float, 2> weight{2.0F, -1.0F};
    std::array<float, 4> out{};
    brt::reference::rms_norm(x, weight, out, RmsNormShape{2, 2}, 0.0F);
    expect_vector_near(out, std::array<float, 4>{1.6970563F, -1.1313709F, 1.2649111F, -1.2649111F});
  }

  {
    const std::array<float, 8> x{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::array<float, 8> zero{};
    std::array<float, 8> one{};
    brt::reference::rope(x, zero, RopeShape{1, 1, 8, 4, 0, 10000.0F, true});
    expect_vector_near(zero, x);
    brt::reference::rope(x, one, RopeShape{1, 1, 8, 4, 1, 10000.0F, true});
    const float c0 = std::cos(1.0F);
    const float s0 = std::sin(1.0F);
    const float c1 = std::cos(0.01F);
    const float s1 = std::sin(0.01F);
    expect_vector_near(one, std::array<float, 8>{
                                1.0F * c0 - 2.0F * s0,
                                1.0F * s0 + 2.0F * c0,
                                3.0F * c1 - 4.0F * s1,
                                3.0F * s1 + 4.0F * c1,
                                5.0F, 6.0F, 7.0F, 8.0F});
  }

  {
    const std::array<float, 4> x_f32{1.0F, -2.0F, 0.5F, 3.0F};
    const std::array<float, 4> w_f32{1.0F, 0.0F, 0.0F, 1.0F};
    std::array<brt::reference::bf16_t, 4> x{};
    std::array<brt::reference::bf16_t, 4> w{};
    std::transform(x_f32.begin(), x_f32.end(), x.begin(), brt::reference::float_to_bf16);
    std::transform(w_f32.begin(), w_f32.end(), w.begin(), brt::reference::float_to_bf16);
    std::array<float, 4> out{};
    brt::reference::bf16_linear(x, w, out, LinearShape{2, 2, 2});
    expect_vector_near(out, x_f32);
    assert(brt::reference::float_to_bf16(1.00390625F).bits == 0x3f80U);
    assert(brt::reference::float_to_bf16(1.01171875F).bits == 0x3f82U);
  }

  {
    const std::array<float, 3> logits{1000.0F, 1001.0F, 999.0F};
    std::array<float, 3> out{};
    brt::reference::softmax(logits, out, SoftmaxShape{1, 3});
    const double denom = std::exp(0.0) + std::exp(1.0) + std::exp(-1.0);
    expect_near(out[0], static_cast<float>(1.0 / denom));
    expect_near(out[1], static_cast<float>(std::exp(1.0) / denom));
    expect_near(out[2], static_cast<float>(std::exp(-1.0) / denom));
  }

  {
    const std::array<float, 6> gate{0.0F, 1.0F, -1.0F, 2.0F, -2.0F, 0.5F};
    const std::array<float, 6> up{2.0F, 3.0F, 4.0F, -1.0F, 0.25F, -0.5F};
    std::array<float, 6> out{};
    brt::reference::swiglu(gate, up, out, SwiGluShape{6});
    std::array<float, 6> expected{};
    for (std::size_t i = 0; i < expected.size(); ++i) {
      expected[i] = silu(gate[i]) * up[i];
    }
    expect_vector_near(out, expected);
  }

  {
    std::mt19937 rng{0xB124};
    std::uniform_real_distribution<float> dist(-3.0F, 3.0F);
    for (int trial = 0; trial < 32; ++trial) {
      std::vector<float> x(64);
      std::vector<float> y(64);
      std::vector<float> out(64);
      std::generate(x.begin(), x.end(), [&] { return dist(rng); });
      std::generate(y.begin(), y.end(), [&] { return dist(rng); });

      brt::reference::add(x, y, out, AddShape{x.size()});
      for (std::size_t i = 0; i < x.size(); ++i) {
        expect_near(out[i], x[i] + y[i]);
      }

      brt::reference::swiglu(x, y, out, SwiGluShape{x.size()});
      for (std::size_t i = 0; i < x.size(); ++i) {
        expect_near(out[i], silu(x[i]) * y[i]);
      }

      brt::reference::softmax(x, out, SoftmaxShape{8, 8});
      for (std::size_t row = 0; row < 8; ++row) {
        const auto row_in = std::span<const float>(x).subspan(row * 8, 8);
        const auto row_out = std::span<const float>(out).subspan(row * 8, 8);
        const float max = *std::max_element(row_in.begin(), row_in.end());
        double sum = 0.0;
        for (const float value : row_in) {
          sum += std::exp(static_cast<double>(value - max));
        }
        for (std::size_t col = 0; col < 8; ++col) {
          expect_near(row_out[col],
                      static_cast<float>(std::exp(static_cast<double>(row_in[col] - max)) / sum));
        }
      }
    }
  }

  {
    std::array<float, 4> out{};
    expect_invalid_argument([&] {
      brt::reference::rms_norm(std::array<float, 4>{}, std::array<float, 2>{}, out,
                               RmsNormShape{2, 2}, -1.0e-5F);
    });
    expect_invalid_argument([&] {
      brt::reference::rope(std::array<float, 4>{}, out, RopeShape{1, 1, 4, 3, 0, 10000.0F, false});
    });
    expect_invalid_argument([&] {
      brt::reference::embedding(std::array<std::int32_t, 2>{0, 2}, std::array<float, 4>{}, out,
                                EmbeddingShape{2, 2, 2});
    });
    expect_vector_near(out, std::array<float, 4>{0.0F, 0.0F, 0.0F, 0.0F});
    expect_invalid_argument([&] {
      brt::reference::add(std::array<float, 3>{}, std::array<float, 4>{}, out, AddShape{4});
    });
  }
}
