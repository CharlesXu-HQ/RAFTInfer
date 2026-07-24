#include "qwen35.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace brt::reference {
namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, const char *message) {
  require(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs,
          message);
  return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char *message) {
  require(rhs <= std::numeric_limits<std::size_t>::max() - lhs, message);
  return lhs + rhs;
}

void require_span(std::size_t actual, std::size_t expected,
                  const char *message) {
  require(actual == expected, message);
}

float sigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }

float silu(float value) { return value / (1.0F + std::exp(-value)); }

float softplus(float value) {
  if (value > 20.0F) {
    return value;
  }
  if (value < -20.0F) {
    return std::exp(value);
  }
  return std::log1p(std::exp(value));
}

std::size_t full_q_size(const FullAttentionReferenceArgs &args) {
  return checked_mul(args.query_heads, args.head_dim,
                     "full attention q size overflow");
}

std::size_t full_kv_size(const FullAttentionReferenceArgs &args) {
  return checked_mul(args.kv_heads, args.head_dim,
                     "full attention kv size overflow");
}

std::size_t full_stride(const FullAttentionReferenceArgs &args) {
  return checked_add(checked_add(full_q_size(args), full_kv_size(args),
                                 "full attention stride overflow"),
                     checked_add(full_kv_size(args), args.hidden_size,
                                 "full attention stride overflow"),
                     "full attention stride overflow");
}

std::size_t delta_key_size(const GatedDeltaReferenceArgs &args) {
  return checked_mul(args.key_heads, args.key_dim,
                     "gated delta key size overflow");
}

std::size_t delta_value_size(const GatedDeltaReferenceArgs &args) {
  return checked_mul(args.value_heads, args.value_dim,
                     "gated delta value size overflow");
}

std::size_t delta_stride(const GatedDeltaReferenceArgs &args) {
  std::size_t stride = checked_add(delta_key_size(args), delta_key_size(args),
                                   "gated delta stride overflow");
  stride = checked_add(stride, delta_value_size(args),
                       "gated delta stride overflow");
  stride = checked_add(stride, args.value_heads, "gated delta stride overflow");
  stride = checked_add(stride, args.value_heads, "gated delta stride overflow");
  return checked_add(stride, args.hidden_size, "gated delta stride overflow");
}

std::size_t delta_conv_dim(const GatedDeltaReferenceArgs &args) {
  return checked_add(checked_add(delta_key_size(args), delta_key_size(args),
                                 "gated delta conv_dim overflow"),
                     delta_value_size(args), "gated delta conv_dim overflow");
}

void validate_full_args(const FullAttentionReferenceArgs &args) {
  require(args.tokens > 0, "full attention tokens must be positive");
  require(args.hidden_size > 0, "full attention hidden_size must be positive");
  require(args.query_heads > 0, "full attention query_heads must be positive");
  require(args.kv_heads > 0, "full attention kv_heads must be positive");
  require(args.head_dim > 0, "full attention head_dim must be positive");
  require(args.rotary_dim <= args.head_dim,
          "full attention rotary_dim exceeds head_dim");
  require(args.rotary_dim % 2 == 0, "full attention rotary_dim must be even");
  require(args.rope_base > 0.0F, "full attention rope_base must be positive");
  require(args.epsilon >= 0.0F, "full attention epsilon must be non-negative");
  require(args.query_heads % args.kv_heads == 0,
          "full attention query_heads must be divisible by kv_heads");
  require(full_q_size(args) == args.hidden_size,
          "full attention query_heads * head_dim must equal hidden_size");
}

void validate_delta_args(const GatedDeltaReferenceArgs &args) {
  require(args.tokens > 0, "gated delta tokens must be positive");
  require(args.hidden_size > 0, "gated delta hidden_size must be positive");
  require(args.key_heads > 0, "gated delta key_heads must be positive");
  require(args.value_heads > 0, "gated delta value_heads must be positive");
  require(args.key_dim > 0, "gated delta key_dim must be positive");
  require(args.value_dim > 0, "gated delta value_dim must be positive");
  require(args.conv_width > 0, "gated delta conv_width must be positive");
  require(args.epsilon >= 0.0F, "gated delta epsilon must be non-negative");
  require(args.value_heads % args.key_heads == 0,
          "gated delta value_heads must be divisible by key_heads");
  require(delta_value_size(args) == args.hidden_size,
          "gated delta value_heads * value_dim must equal hidden_size");
}

void rms_norm_with_multiplier(std::span<const float> input,
                              std::span<const float> weight,
                              std::span<float> output, std::size_t rows,
                              std::size_t cols, float epsilon,
                              float weight_base) {
  require(epsilon >= 0.0F, "rms_norm epsilon must be non-negative");
  require(rows > 0, "rms_norm rows must be positive");
  require(cols > 0, "rms_norm cols must be positive");
  const std::size_t total = checked_mul(rows, cols, "rms_norm shape overflow");
  require_span(input.size(), total, "rms_norm input span does not match shape");
  require_span(weight.size(), cols,
               "rms_norm weight span does not match shape");
  require_span(output.size(), total,
               "rms_norm output span does not match shape");

  for (std::size_t row = 0; row < rows; ++row) {
    double square_sum = 0.0;
    for (std::size_t col = 0; col < cols; ++col) {
      const double value = input[row * cols + col];
      square_sum += value * value;
    }
    const float scale =
        1.0F /
        std::sqrt(static_cast<float>(square_sum / static_cast<double>(cols)) +
                  epsilon);
    for (std::size_t col = 0; col < cols; ++col) {
      output[row * cols + col] =
          input[row * cols + col] * scale * (weight_base + weight[col]);
    }
  }
}

void l2_normalize(std::span<float> values, float epsilon) {
  double square_sum = 0.0;
  for (const float value : values) {
    square_sum += static_cast<double>(value) * static_cast<double>(value);
  }
  const float scale =
      1.0F / std::sqrt(static_cast<float>(square_sum) + epsilon);
  for (float &value : values) {
    value *= scale;
  }
}

void apply_partial_rope(std::span<float> vector, std::size_t rotary_dim,
                        std::size_t position, float rope_base) {
  const std::size_t pair_count = rotary_dim / 2;
  for (std::size_t pair = 0; pair < pair_count; ++pair) {
    const double exponent =
        static_cast<double>(2 * pair) / static_cast<double>(rotary_dim);
    const double theta =
        static_cast<double>(position) / std::pow(rope_base, exponent);
    const float cos_theta = static_cast<float>(std::cos(theta));
    const float sin_theta = static_cast<float>(std::sin(theta));
    const std::size_t first = pair;
    const std::size_t second = pair + pair_count;
    const float x0 = vector[first];
    const float x1 = vector[second];
    vector[first] = x0 * cos_theta - x1 * sin_theta;
    vector[second] = x0 * sin_theta + x1 * cos_theta;
  }
}

float recurrent_at(const GatedDeltaReferenceArgs &args,
                   const GatedDeltaReferenceState &state,
                   std::size_t value_head, std::size_t key_index,
                   std::size_t value_index) {
  return state
      .recurrent[(value_head * args.key_dim + key_index) * args.value_dim +
                 value_index];
}

float &recurrent_at(const GatedDeltaReferenceArgs &args,
                    GatedDeltaReferenceState &state, std::size_t value_head,
                    std::size_t key_index, std::size_t value_index) {
  return state
      .recurrent[(value_head * args.key_dim + key_index) * args.value_dim +
                 value_index];
}

} // namespace

FullAttentionReferenceArgs
FullAttentionReferenceArgs::from_config(const model::Qwen35Config &config,
                                        std::size_t tokens,
                                        std::size_t position_offset) {
  return FullAttentionReferenceArgs{
      .tokens = tokens,
      .hidden_size = config.hidden_size,
      .query_heads = config.full_attention_head_count,
      .kv_heads = config.full_attention_kv_head_count,
      .head_dim = config.full_attention_head_dimension,
      .rotary_dim = config.rotary_dimension,
      .position_offset = position_offset,
      .rope_base = config.rope_frequency_base,
      .epsilon = config.rms_norm_epsilon};
}

GatedDeltaReferenceArgs
GatedDeltaReferenceArgs::from_config(const model::Qwen35Config &config,
                                     std::size_t tokens) {
  return GatedDeltaReferenceArgs{.tokens = tokens,
                                 .hidden_size = config.hidden_size,
                                 .key_heads = config.linear_key_head_count,
                                 .value_heads = config.linear_value_head_count,
                                 .key_dim = config.linear_head_dimension,
                                 .value_dim = config.linear_head_dimension,
                                 .conv_width = config.linear_convolution_width,
                                 .epsilon = config.rms_norm_epsilon};
}

GatedDeltaReferenceState::GatedDeltaReferenceState(
    const GatedDeltaReferenceArgs &args) {
  validate_delta_args(args);
  convolution.assign(
      checked_mul(delta_conv_dim(args), args.conv_width - 1,
                  "gated delta convolution state shape overflow"),
      0.0F);
  recurrent.assign(checked_mul(delta_value_size(args), args.key_dim,
                               "gated delta recurrent shape overflow"),
                   0.0F);
}

void qwen35_rms_norm(std::span<const float> input,
                     std::span<const float> weight, std::span<float> output,
                     std::size_t rows, std::size_t cols, float epsilon) {
  rms_norm_with_multiplier(input, weight, output, rows, cols, epsilon, 1.0F);
}

void qwen35_gated_full_attention(std::span<const float> input,
                                 std::span<float> output,
                                 FullAttentionReferenceWeights weights,
                                 const FullAttentionReferenceArgs &args) {
  validate_full_args(args);
  const std::size_t q_size = full_q_size(args);
  const std::size_t kv_size = full_kv_size(args);
  const std::size_t token_stride = full_stride(args);
  require_span(
      input.size(),
      checked_mul(args.tokens, token_stride, "full attention input overflow"),
      "full attention input span does not match shape");
  require_span(output.size(),
               checked_mul(args.tokens, args.hidden_size,
                           "full attention output overflow"),
               "full attention output span does not match shape");
  require_span(weights.query_norm_weight.size(), args.head_dim,
               "full attention query_norm_weight span does not match head_dim");
  require_span(weights.key_norm_weight.size(), args.head_dim,
               "full attention key_norm_weight span does not match head_dim");
  require_span(weights.output_weight.size(),
               checked_mul(args.hidden_size, args.hidden_size,
                           "full attention output_weight overflow"),
               "full attention output_weight span does not match shape");

  std::vector<float> queries(
      checked_mul(args.tokens, q_size, "full attention q buffer overflow"));
  std::vector<float> keys(
      checked_mul(args.tokens, kv_size, "full attention k buffer overflow"));
  std::vector<float> values(
      checked_mul(args.tokens, kv_size, "full attention v buffer overflow"));

  for (std::size_t token = 0; token < args.tokens; ++token) {
    const std::size_t input_base = token * token_stride;
    const auto q_in = input.subspan(input_base, q_size);
    const auto k_in = input.subspan(input_base + q_size, kv_size);
    const auto v_in = input.subspan(input_base + q_size + kv_size, kv_size);
    std::copy(q_in.begin(), q_in.end(),
              queries.begin() + static_cast<std::ptrdiff_t>(token * q_size));
    std::copy(k_in.begin(), k_in.end(),
              keys.begin() + static_cast<std::ptrdiff_t>(token * kv_size));
    std::copy(v_in.begin(), v_in.end(),
              values.begin() + static_cast<std::ptrdiff_t>(token * kv_size));

    for (std::size_t head = 0; head < args.query_heads; ++head) {
      auto q = std::span<float>(queries).subspan(
          token * q_size + head * args.head_dim, args.head_dim);
      rms_norm_with_multiplier(q, weights.query_norm_weight, q, 1,
                               args.head_dim, args.epsilon, 1.0F);
      apply_partial_rope(q, args.rotary_dim, args.position_offset + token,
                         args.rope_base);
    }
    for (std::size_t head = 0; head < args.kv_heads; ++head) {
      auto k = std::span<float>(keys).subspan(
          token * kv_size + head * args.head_dim, args.head_dim);
      rms_norm_with_multiplier(k, weights.key_norm_weight, k, 1, args.head_dim,
                               args.epsilon, 1.0F);
      apply_partial_rope(k, args.rotary_dim, args.position_offset + token,
                         args.rope_base);
    }
  }

  std::vector<float> attention(args.hidden_size, 0.0F);
  std::vector<float> logits(args.tokens, 0.0F);
  std::vector<float> probabilities(args.tokens, 0.0F);
  for (std::size_t token = 0; token < args.tokens; ++token) {
    std::fill(attention.begin(), attention.end(), 0.0F);
    for (std::size_t query_head = 0; query_head < args.query_heads;
         ++query_head) {
      const std::size_t kv_head =
          query_head / (args.query_heads / args.kv_heads);
      const auto query = std::span<const float>(queries).subspan(
          token * q_size + query_head * args.head_dim, args.head_dim);

      float max_logit = -std::numeric_limits<float>::infinity();
      for (std::size_t key_token = 0; key_token <= token; ++key_token) {
        const auto key = std::span<const float>(keys).subspan(
            key_token * kv_size + kv_head * args.head_dim, args.head_dim);
        double dot = 0.0;
        for (std::size_t dim = 0; dim < args.head_dim; ++dim) {
          dot +=
              static_cast<double>(query[dim]) * static_cast<double>(key[dim]);
        }
        const float logit = static_cast<float>(
            dot / std::sqrt(static_cast<double>(args.head_dim)));
        logits[key_token] = logit;
        max_logit = std::max(max_logit, logit);
      }

      double probability_sum = 0.0;
      for (std::size_t key_token = 0; key_token <= token; ++key_token) {
        const double shifted = static_cast<double>(logits[key_token]) -
                               static_cast<double>(max_logit);
        probabilities[key_token] = static_cast<float>(std::exp(shifted));
        probability_sum += probabilities[key_token];
      }
      for (std::size_t key_token = 0; key_token <= token; ++key_token) {
        probabilities[key_token] = static_cast<float>(
            static_cast<double>(probabilities[key_token]) / probability_sum);
      }

      auto head_output = std::span<float>(attention).subspan(
          query_head * args.head_dim, args.head_dim);
      for (std::size_t key_token = 0; key_token <= token; ++key_token) {
        const auto value = std::span<const float>(values).subspan(
            key_token * kv_size + kv_head * args.head_dim, args.head_dim);
        for (std::size_t dim = 0; dim < args.head_dim; ++dim) {
          head_output[dim] += probabilities[key_token] * value[dim];
        }
      }
    }

    const auto gate = input.subspan(token * token_stride + q_size + 2 * kv_size,
                                    args.hidden_size);
    for (std::size_t col = 0; col < args.hidden_size; ++col) {
      attention[col] *= sigmoid(gate[col]);
    }

    for (std::size_t out_col = 0; out_col < args.hidden_size; ++out_col) {
      double sum = 0.0;
      for (std::size_t in_col = 0; in_col < args.hidden_size; ++in_col) {
        sum += static_cast<double>(attention[in_col]) *
               static_cast<double>(
                   weights.output_weight[in_col * args.hidden_size + out_col]);
      }
      output[token * args.hidden_size + out_col] = static_cast<float>(sum);
    }
  }
}

void qwen35_gated_delta_step(std::span<const float> input,
                             std::span<float> output,
                             GatedDeltaReferenceWeights weights,
                             const GatedDeltaReferenceArgs &args,
                             GatedDeltaReferenceState &state) {
  validate_delta_args(args);
  const std::size_t key_size = delta_key_size(args);
  const std::size_t value_size = delta_value_size(args);
  const std::size_t token_stride = delta_stride(args);
  const std::size_t conv_dim = delta_conv_dim(args);
  require_span(input.size(), token_stride,
               "gated delta step input span does not match shape");
  require_span(output.size(), args.hidden_size,
               "gated delta step output span does not match shape");
  require_span(weights.conv_weight.size(),
               checked_mul(conv_dim, args.conv_width,
                           "gated delta conv_weight overflow"),
               "gated delta conv_weight span does not match shape");
  require_span(weights.a_log.size(), args.value_heads,
               "gated delta a_log span does not match shape");
  require_span(weights.dt_bias.size(), args.value_heads,
               "gated delta dt_bias span does not match shape");
  require_span(weights.output_norm_weight.size(), args.value_dim,
               "gated delta output_norm_weight span does not match value_dim");
  require_span(state.convolution.size(),
               checked_mul(conv_dim, args.conv_width - 1,
                           "gated delta convolution state shape overflow"),
               "gated delta convolution state span does not match shape");
  require_span(state.recurrent.size(),
               checked_mul(value_size, args.key_dim,
                           "gated delta recurrent state shape overflow"),
               "gated delta recurrent state span does not match shape");

  std::vector<float> convolved(conv_dim, 0.0F);
  for (std::size_t channel = 0; channel < conv_dim; ++channel) {
    float sum =
        input[channel] *
        weights.conv_weight[channel * args.conv_width + args.conv_width - 1];
    for (std::size_t history = 0; history + 1 < args.conv_width; ++history) {
      sum += state.convolution[channel * (args.conv_width - 1) + history] *
             weights.conv_weight[channel * args.conv_width + history];
    }
    convolved[channel] = silu(sum);
  }

  if (args.conv_width > 1) {
    for (std::size_t channel = 0; channel < conv_dim; ++channel) {
      const std::size_t base = channel * (args.conv_width - 1);
      for (std::size_t history = 0; history + 2 < args.conv_width; ++history) {
        state.convolution[base + history] =
            state.convolution[base + history + 1];
      }
      state.convolution[base + args.conv_width - 2] = input[channel];
    }
  }

  auto q = std::span<float>(convolved).subspan(0, key_size);
  auto k = std::span<float>(convolved).subspan(key_size, key_size);
  auto v = std::span<float>(convolved).subspan(2 * key_size, value_size);
  const auto b = input.subspan(conv_dim, args.value_heads);
  const auto a = input.subspan(conv_dim + args.value_heads, args.value_heads);
  const auto z = input.subspan(2 * key_size + value_size + 2 * args.value_heads,
                               args.hidden_size);

  const float query_scale = 1.0F / std::sqrt(static_cast<float>(args.key_dim));
  for (std::size_t head = 0; head < args.key_heads; ++head) {
    auto q_head = q.subspan(head * args.key_dim, args.key_dim);
    auto k_head = k.subspan(head * args.key_dim, args.key_dim);
    l2_normalize(q_head, 1.0e-6F);
    l2_normalize(k_head, 1.0e-6F);
    for (float &value : q_head) {
      value *= query_scale;
    }
  }

  std::vector<float> raw(value_size, 0.0F);
  const std::size_t repeat = args.value_heads / args.key_heads;
  for (std::size_t value_head = 0; value_head < args.value_heads;
       ++value_head) {
    const std::size_t key_head = value_head / repeat;
    const auto q_head = q.subspan(key_head * args.key_dim, args.key_dim);
    const auto k_head = k.subspan(key_head * args.key_dim, args.key_dim);
    auto v_head = v.subspan(value_head * args.value_dim, args.value_dim);
    const float beta = sigmoid(b[value_head]);
    const float decay =
        std::exp(-std::exp(weights.a_log[value_head]) *
                 softplus(a[value_head] + weights.dt_bias[value_head]));

    for (std::size_t key_index = 0; key_index < args.key_dim; ++key_index) {
      for (std::size_t value_index = 0; value_index < args.value_dim;
           ++value_index) {
        recurrent_at(args, state, value_head, key_index, value_index) *= decay;
      }
    }

    std::array<float, 64> stack_delta{};
    std::vector<float> heap_delta;
    std::span<float> delta;
    if (args.value_dim <= stack_delta.size()) {
      delta = std::span<float>(stack_delta).first(args.value_dim);
    } else {
      heap_delta.assign(args.value_dim, 0.0F);
      delta = heap_delta;
    }

    for (std::size_t value_index = 0; value_index < args.value_dim;
         ++value_index) {
      double kv_mem = 0.0;
      for (std::size_t key_index = 0; key_index < args.key_dim; ++key_index) {
        kv_mem += static_cast<double>(recurrent_at(args, state, value_head,
                                                   key_index, value_index)) *
                  static_cast<double>(k_head[key_index]);
      }
      delta[value_index] =
          (v_head[value_index] - static_cast<float>(kv_mem)) * beta;
    }

    for (std::size_t key_index = 0; key_index < args.key_dim; ++key_index) {
      for (std::size_t value_index = 0; value_index < args.value_dim;
           ++value_index) {
        recurrent_at(args, state, value_head, key_index, value_index) +=
            k_head[key_index] * delta[value_index];
      }
    }

    for (std::size_t value_index = 0; value_index < args.value_dim;
         ++value_index) {
      double sum = 0.0;
      for (std::size_t key_index = 0; key_index < args.key_dim; ++key_index) {
        sum += static_cast<double>(recurrent_at(args, state, value_head,
                                                key_index, value_index)) *
               static_cast<double>(q_head[key_index]);
      }
      raw[value_head * args.value_dim + value_index] = static_cast<float>(sum);
    }
  }

  rms_norm_with_multiplier(raw, weights.output_norm_weight, output,
                           args.value_heads, args.value_dim, args.epsilon,
                           0.0F);
  for (std::size_t index = 0; index < args.hidden_size; ++index) {
    output[index] *= silu(z[index]);
  }
}

void qwen35_gated_delta_prefill(std::span<const float> input,
                                std::span<float> output,
                                GatedDeltaReferenceWeights weights,
                                const GatedDeltaReferenceArgs &args,
                                GatedDeltaReferenceState &state) {
  validate_delta_args(args);
  const std::size_t token_stride = delta_stride(args);
  require_span(
      input.size(),
      checked_mul(args.tokens, token_stride, "gated delta input overflow"),
      "gated delta prefill input span does not match shape");
  require_span(
      output.size(),
      checked_mul(args.tokens, args.hidden_size, "gated delta output overflow"),
      "gated delta prefill output span does not match shape");
  for (std::size_t token = 0; token < args.tokens; ++token) {
    qwen35_gated_delta_step(
        input.subspan(token * token_stride, token_stride),
        output.subspan(token * args.hidden_size, args.hidden_size), weights,
        args, state);
  }
}

} // namespace brt::reference
