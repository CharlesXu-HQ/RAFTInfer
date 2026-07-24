#include "../reference/bf16.hpp"
#include "../reference/operators.hpp"

#include "assert_enabled.hpp"

#include <algorithm>
#include <array>
#include <bit>
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

void expect_vector_equal(std::span<const float> actual, std::span<const float> expected) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    assert(actual[i] == expected[i]);
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

float float_from_bits(std::uint32_t bits) {
  return std::bit_cast<float>(bits);
}

void expect_bf16_bits(float value, std::uint16_t bits) {
  assert(brt::reference::float_to_bf16(value).bits == bits);
}

void expect_bf16_nan(std::uint32_t f32_bits) {
  const auto bf16 = brt::reference::float_to_bf16(float_from_bits(f32_bits));
  assert((bf16.bits & 0x7F80U) == 0x7F80U);
  assert((bf16.bits & 0x007FU) != 0);
}

std::vector<float> scalar_rope(std::span<const float> input, brt::reference::RopeShape shape) {
  std::vector<float> expected(input.begin(), input.end());
  const std::size_t vectors = shape.tokens * shape.heads;
  const std::size_t pair_count = shape.rotary_dim / 2;
  for (std::size_t vector = 0; vector < vectors; ++vector) {
    const std::size_t base_index = vector * shape.head_dim;
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      const double exponent = static_cast<double>(2 * pair) / static_cast<double>(shape.rotary_dim);
      const double theta = static_cast<double>(shape.position) / std::pow(shape.base, exponent);
      const float c = static_cast<float>(std::cos(theta));
      const float s = static_cast<float>(std::sin(theta));
      const std::size_t first = base_index + (shape.interleaved ? 2 * pair : pair);
      const std::size_t second = base_index + (shape.interleaved ? 2 * pair + 1 : pair + pair_count);
      const float x0 = input[first];
      const float x1 = input[second];
      expected[first] = x0 * c - x1 * s;
      expected[second] = x0 * s + x1 * c;
    }
  }
  return expected;
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
    expect_vector_near(one, scalar_rope(x, RopeShape{1, 1, 8, 4, 1, 10000.0F, true}));
  }

  {
    const std::array<float, 6> x{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::array<float, 6> out{};
    brt::reference::rope(x, out, RopeShape{1, 1, 6, 4, 3, 100.0F, false});
    expect_vector_near(out, scalar_rope(x, RopeShape{1, 1, 6, 4, 3, 100.0F, false}));
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
    expect_bf16_bits(1.00390625F, 0x3F80U);
    expect_bf16_bits(1.01171875F, 0x3F82U);
    expect_bf16_bits(std::numeric_limits<float>::infinity(), 0x7F80U);
    expect_bf16_bits(-std::numeric_limits<float>::infinity(), 0xFF80U);
    expect_bf16_nan(0x7FC00001U);
    expect_bf16_nan(0x7FA00001U);
    expect_bf16_nan(0xFFFF8000U);
    expect_bf16_nan(0xFF800001U);
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
    std::uniform_int_distribution<int> token_dist(0, 6);
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
          sum += std::exp(static_cast<double>(value) - static_cast<double>(max));
        }
        for (std::size_t col = 0; col < 8; ++col) {
          expect_near(row_out[col],
                      static_cast<float>(std::exp(static_cast<double>(row_in[col]) -
                                                  static_cast<double>(max)) /
                                         sum));
        }
      }

      std::array<float, 8> weight{};
      std::copy_n(y.begin(), weight.size(), weight.begin());
      brt::reference::rms_norm(x, weight, out, RmsNormShape{8, 8}, 1.0e-5F);
      for (std::size_t row = 0; row < 8; ++row) {
        double mean_square = 0.0;
        for (std::size_t col = 0; col < 8; ++col) {
          const double value = static_cast<double>(x[row * 8 + col]);
          mean_square += value * value;
        }
        mean_square /= 8.0;
        const double scale = 1.0 / std::sqrt(mean_square + 1.0e-5);
        for (std::size_t col = 0; col < 8; ++col) {
          expect_near(out[row * 8 + col],
                      static_cast<float>(static_cast<double>(x[row * 8 + col]) *
                                         scale * static_cast<double>(weight[col])));
        }
      }

      brt::reference::rope(x, out, RopeShape{2, 2, 16, 8, static_cast<std::size_t>(trial + 1),
                                             10000.0F, (trial % 2) == 0});
      expect_vector_near(out, scalar_rope(x, RopeShape{2, 2, 16, 8,
                                                       static_cast<std::size_t>(trial + 1),
                                                       10000.0F, (trial % 2) == 0}));

      std::array<float, 15> linear_input_f32{};
      std::array<float, 20> linear_weight_f32{};
      std::array<brt::reference::bf16_t, 15> linear_input{};
      std::array<brt::reference::bf16_t, 20> linear_weight{};
      std::copy_n(x.begin(), linear_input_f32.size(), linear_input_f32.begin());
      std::copy_n(y.begin(), linear_weight_f32.size(), linear_weight_f32.begin());
      std::transform(linear_input_f32.begin(), linear_input_f32.end(), linear_input.begin(),
                     brt::reference::float_to_bf16);
      std::transform(linear_weight_f32.begin(), linear_weight_f32.end(), linear_weight.begin(),
                     brt::reference::float_to_bf16);
      std::array<float, 12> linear_out{};
      brt::reference::bf16_linear(linear_input, linear_weight, linear_out, LinearShape{3, 5, 4});
      for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
          float sum = 0.0F;
          for (std::size_t k = 0; k < 5; ++k) {
            sum += brt::reference::bf16_to_float(linear_input[row * 5 + k]) *
                   brt::reference::bf16_to_float(linear_weight[k * 4 + col]);
          }
          expect_near(linear_out[row * 4 + col], sum);
        }
      }

      std::array<std::int32_t, 6> tokens{};
      std::array<float, 35> table{};
      std::array<float, 30> embedding_out{};
      std::generate(tokens.begin(), tokens.end(), [&] {
        return static_cast<std::int32_t>(token_dist(rng));
      });
      std::copy_n(x.begin(), table.size(), table.begin());
      brt::reference::embedding(tokens, table, embedding_out, EmbeddingShape{6, 5, 7});
      for (std::size_t row = 0; row < tokens.size(); ++row) {
        for (std::size_t col = 0; col < 5; ++col) {
          expect_near(embedding_out[row * 5 + col],
                      table[static_cast<std::size_t>(tokens[row]) * 5 + col]);
        }
      }

      std::vector<float> arg_values(x.begin(), x.end());
      arg_values[10] = 9.0F;
      arg_values[20] = 9.0F;
      assert(brt::reference::argmax(arg_values) == 10);
    }
  }

  {
    const std::array<float, 4> sentinel4{7.0F, 8.0F, 9.0F, 10.0F};
    std::array<float, 4> out{};
    const auto max_size = std::numeric_limits<std::size_t>::max();

    brt::reference::add(std::span<const float>{}, std::span<const float>{}, std::span<float>{},
                        AddShape{0});
    brt::reference::swiglu(std::span<const float>{}, std::span<const float>{}, std::span<float>{},
                           SwiGluShape{0});
    brt::reference::bf16_linear(std::span<const brt::reference::bf16_t>{},
                                std::array<brt::reference::bf16_t, 6>{}, std::span<float>{},
                                LinearShape{0, 3, 2});
    brt::reference::embedding(std::span<const std::int32_t>{}, std::array<float, 4>{},
                              std::span<float>{}, EmbeddingShape{0, 2, 2});
    brt::reference::rope(std::span<const float>{}, std::span<float>{},
                         RopeShape{0, 1, 4, 4, 0, 10000.0F, true});
    brt::reference::softmax(std::span<const float>{}, std::span<float>{}, SoftmaxShape{0, 3});

    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::add(std::array<float, 3>{}, std::array<float, 4>{}, out, AddShape{4});
    });
    expect_vector_equal(out, sentinel4);

    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::rms_norm(std::array<float, 4>{}, std::array<float, 2>{}, out,
                               RmsNormShape{2, 2}, -1.0e-5F);
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::rms_norm(std::span<const float>{}, std::span<const float>{}, out,
                               RmsNormShape{max_size, 2}, 0.0F);
    });
    expect_vector_equal(out, sentinel4);
    expect_invalid_argument([&] {
      brt::reference::rms_norm(std::span<const float>{}, std::span<const float>{},
                               std::span<float>{}, RmsNormShape{1, 0}, 0.0F);
    });
    brt::reference::rms_norm(std::span<const float>{}, std::array<float, 3>{},
                             std::span<float>{}, RmsNormShape{0, 3}, 1.0e-5F);

    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::rope(std::array<float, 4>{}, out, RopeShape{1, 1, 4, 6, 0, 10000.0F, false});
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::rope(std::array<float, 4>{}, out, RopeShape{1, 1, 4, 4, 0, 0.0F, false});
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::rope(std::array<float, 4>{}, out, RopeShape{1, 1, 4, 3, 0, 10000.0F, false});
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::rope(std::span<const float>{}, out,
                           RopeShape{max_size, 2, 4, 4, 0, 10000.0F, false});
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::rope(std::span<const float>{}, out,
                           RopeShape{max_size / 2U + 1U, 1, 2, 2, 0, 10000.0F, false});
    });
    expect_vector_equal(out, sentinel4);

    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::bf16_linear(std::span<const brt::reference::bf16_t>{},
                                  std::span<const brt::reference::bf16_t>{}, out,
                                  LinearShape{max_size, 2, 1});
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::bf16_linear(std::span<const brt::reference::bf16_t>{},
                                  std::span<const brt::reference::bf16_t>{}, out,
                                  LinearShape{max_size, 1, 2});
    });
    expect_vector_equal(out, sentinel4);

    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::embedding(std::array<std::int32_t, 2>{0, -1}, std::array<float, 4>{}, out,
                                EmbeddingShape{2, 2, 2});
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::embedding(std::array<std::int32_t, 2>{0, 2}, std::array<float, 4>{}, out,
                                EmbeddingShape{2, 2, 2});
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::embedding(std::span<const std::int32_t>{}, std::span<const float>{}, out,
                                EmbeddingShape{1, 2, max_size});
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::embedding(std::span<const std::int32_t>{}, std::span<const float>{}, out,
                                EmbeddingShape{max_size, 2, 2});
    });
    expect_vector_equal(out, sentinel4);

    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::softmax(std::span<const float>{}, out, SoftmaxShape{1, 0});
    });
    expect_vector_equal(out, sentinel4);
    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::softmax(std::span<const float>{}, out, SoftmaxShape{max_size, 2});
    });
    expect_vector_equal(out, sentinel4);

    out = sentinel4;
    expect_invalid_argument([&] {
      brt::reference::swiglu(std::array<float, 3>{}, std::array<float, 4>{}, out, SwiGluShape{4});
    });
    expect_vector_equal(out, sentinel4);

    expect_invalid_argument([&] {
      static_cast<void>(brt::reference::argmax(std::span<const float>{}));
    });
  }
}
