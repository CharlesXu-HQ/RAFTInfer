#include "operators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace raftinfer::reference {
namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, const char* message) {
  require(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs, message);
  return lhs * rhs;
}

void require_span(std::size_t actual, std::size_t expected, const char* message) {
  require(actual == expected, message);
}

float silu(float value) {
  return value / (1.0F + std::exp(-value));
}

}  // namespace

void add(
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::span<float> output,
    AddShape shape) {
  require_span(lhs.size(), shape.elements, "add lhs span does not match shape");
  require_span(rhs.size(), shape.elements, "add rhs span does not match shape");
  require_span(output.size(), shape.elements, "add output span does not match shape");

  for (std::size_t i = 0; i < shape.elements; ++i) {
    output[i] = lhs[i] + rhs[i];
  }
}

std::size_t argmax(std::span<const float> values) {
  require(!values.empty(), "argmax input must not be empty");

  std::size_t best = 0;
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return best;
}

void bf16_linear(
    std::span<const bf16_t> input,
    std::span<const bf16_t> weight,
    std::span<float> output,
    LinearShape shape) {
  const std::size_t input_size =
      checked_mul(shape.batch, shape.in_features, "linear input shape overflow");
  const std::size_t weight_size =
      checked_mul(shape.in_features, shape.out_features, "linear weight shape overflow");
  const std::size_t output_size =
      checked_mul(shape.batch, shape.out_features, "linear output shape overflow");
  require_span(input.size(), input_size, "linear input span does not match shape");
  require_span(weight.size(), weight_size, "linear weight span does not match shape");
  require_span(output.size(), output_size, "linear output span does not match shape");

  for (std::size_t row = 0; row < shape.batch; ++row) {
    for (std::size_t col = 0; col < shape.out_features; ++col) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < shape.in_features; ++k) {
        sum += bf16_to_float(input[row * shape.in_features + k]) *
               bf16_to_float(weight[k * shape.out_features + col]);
      }
      output[row * shape.out_features + col] = sum;
    }
  }
}

void embedding(
    std::span<const std::int32_t> tokens,
    std::span<const float> table,
    std::span<float> output,
    EmbeddingShape shape) {
  const std::size_t table_size =
      checked_mul(shape.vocab_size, shape.embedding_dim, "embedding table shape overflow");
  const std::size_t output_size =
      checked_mul(shape.tokens, shape.embedding_dim, "embedding output shape overflow");
  require_span(tokens.size(), shape.tokens, "embedding token span does not match shape");
  require_span(table.size(), table_size, "embedding table span does not match shape");
  require_span(output.size(), output_size, "embedding output span does not match shape");

  for (const std::int32_t token : tokens) {
    require(token >= 0, "embedding token id is negative");
    require(static_cast<std::size_t>(token) < shape.vocab_size,
            "embedding token id is out of range");
  }

  for (std::size_t token_index = 0; token_index < shape.tokens; ++token_index) {
    const auto token = static_cast<std::size_t>(tokens[token_index]);
    for (std::size_t col = 0; col < shape.embedding_dim; ++col) {
      output[token_index * shape.embedding_dim + col] = table[token * shape.embedding_dim + col];
    }
  }
}

void rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    RmsNormShape shape,
    float epsilon) {
  require(epsilon >= 0.0F, "rms_norm epsilon must be non-negative");
  require(shape.cols > 0, "rms_norm cols must be positive");
  const std::size_t input_size = checked_mul(shape.rows, shape.cols, "rms_norm shape overflow");
  require_span(input.size(), input_size, "rms_norm input span does not match shape");
  require_span(weight.size(), shape.cols, "rms_norm weight span does not match shape");
  require_span(output.size(), input_size, "rms_norm output span does not match shape");

  for (std::size_t row = 0; row < shape.rows; ++row) {
    double mean_square = 0.0;
    for (std::size_t col = 0; col < shape.cols; ++col) {
      const float value = input[row * shape.cols + col];
      mean_square += static_cast<double>(value) * static_cast<double>(value);
    }
    mean_square /= static_cast<double>(shape.cols);
    const float scale = 1.0F / std::sqrt(static_cast<float>(mean_square) + epsilon);
    for (std::size_t col = 0; col < shape.cols; ++col) {
      output[row * shape.cols + col] = input[row * shape.cols + col] * scale * weight[col];
    }
  }
}

void rope(
    std::span<const float> input,
    std::span<float> output,
    RopeShape shape) {
  require(shape.rotary_dim <= shape.head_dim, "rope rotary_dim exceeds head_dim");
  require(shape.rotary_dim % 2 == 0, "rope rotary_dim must be even");
  require(shape.base > 0.0F, "rope base must be positive");
  const std::size_t vectors = checked_mul(shape.tokens, shape.heads, "rope vector shape overflow");
  const std::size_t total = checked_mul(vectors, shape.head_dim, "rope input shape overflow");
  require_span(input.size(), total, "rope input span does not match shape");
  require_span(output.size(), total, "rope output span does not match shape");

  std::copy(input.begin(), input.end(), output.begin());

  const std::size_t pair_count = shape.rotary_dim / 2;
  for (std::size_t vector = 0; vector < vectors; ++vector) {
    const std::size_t base_index = vector * shape.head_dim;
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      const double exponent = static_cast<double>(2 * pair) / static_cast<double>(shape.rotary_dim);
      const double theta = static_cast<double>(shape.position) / std::pow(shape.base, exponent);
      const float cos_theta = static_cast<float>(std::cos(theta));
      const float sin_theta = static_cast<float>(std::sin(theta));

      const std::size_t first = base_index + (shape.interleaved ? 2 * pair : pair);
      const std::size_t second = base_index + (shape.interleaved ? 2 * pair + 1 : pair + pair_count);
      const float x0 = input[first];
      const float x1 = input[second];
      output[first] = x0 * cos_theta - x1 * sin_theta;
      output[second] = x0 * sin_theta + x1 * cos_theta;
    }
  }
}

void softmax(
    std::span<const float> input,
    std::span<float> output,
    SoftmaxShape shape) {
  const std::size_t total = checked_mul(shape.rows, shape.cols, "softmax shape overflow");
  require(shape.cols > 0, "softmax cols must be positive");
  require_span(input.size(), total, "softmax input span does not match shape");
  require_span(output.size(), total, "softmax output span does not match shape");

  for (std::size_t row = 0; row < shape.rows; ++row) {
    const std::size_t row_offset = row * shape.cols;
    float max = input[row_offset];
    for (std::size_t col = 1; col < shape.cols; ++col) {
      max = std::max(max, input[row_offset + col]);
    }

    double sum = 0.0;
    for (std::size_t col = 0; col < shape.cols; ++col) {
      const double shifted = static_cast<double>(input[row_offset + col]) - static_cast<double>(max);
      sum += std::exp(shifted);
    }
    for (std::size_t col = 0; col < shape.cols; ++col) {
      const double shifted = static_cast<double>(input[row_offset + col]) - static_cast<double>(max);
      output[row_offset + col] = static_cast<float>(std::exp(shifted) / sum);
    }
  }
}

void swiglu(
    std::span<const float> gate,
    std::span<const float> up,
    std::span<float> output,
    SwiGluShape shape) {
  require_span(gate.size(), shape.elements, "swiglu gate span does not match shape");
  require_span(up.size(), shape.elements, "swiglu up span does not match shape");
  require_span(output.size(), shape.elements, "swiglu output span does not match shape");

  for (std::size_t i = 0; i < shape.elements; ++i) {
    output[i] = silu(gate[i]) * up[i];
  }
}

}  // namespace raftinfer::reference
