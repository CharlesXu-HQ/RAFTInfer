#include "qwen35_executor.hpp"

#include "bf16.hpp"
#include "operators.hpp"
#include "qwen35.hpp"

#include "../model/model.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace brt::reference {
namespace {

constexpr std::uint32_t kF32TensorType = 0;
constexpr std::uint32_t kF16TensorType = 1;
constexpr std::uint32_t kBf16TensorType = 30;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

float f16_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t mantissa = value & 0x03ffU;
  std::uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int unbiased_exponent = -14;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        --unbiased_exponent;
      }
      mantissa &= 0x03ffU;
      bits = sign |
             (static_cast<std::uint32_t>(unbiased_exponent + 127) << 23U) |
             (mantissa << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    exponent = exponent - 15U + 127U;
    bits = sign | (exponent << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

std::vector<float> tensor_f32(const model::Model &model,
                              const gguf::TensorInfo &tensor) {
  require(tensor.type == kF32TensorType || tensor.type == kF16TensorType ||
              tensor.type == kBf16TensorType,
          "CPU Qwen3.5 reference requires F32, F16, or BF16 tensors");
  const auto payload = model.tensor_payload(tensor);
  if (tensor.type == kF32TensorType) {
    require(payload.size() % sizeof(float) == 0,
            "CPU Qwen3.5 F32 tensor payload is not float-aligned");
    std::vector<float> result(payload.size() / sizeof(float));
    std::memcpy(result.data(), payload.data(), payload.size());
    return result;
  }
  require(payload.size() % sizeof(std::uint16_t) == 0,
          "CPU Qwen3.5 tensor payload is not 16-bit aligned");
  std::vector<float> result(payload.size() / sizeof(std::uint16_t));
  for (std::size_t index = 0; index < result.size(); ++index) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, payload.data() + index * sizeof(bits), sizeof(bits));
    result[index] = tensor.type == kBf16TensorType ? bf16_to_float(bf16_t{bits})
                                                   : f16_to_float(bits);
  }
  return result;
}

std::vector<float> linear(std::span<const float> input, std::size_t rows,
                          std::size_t input_width,
                          std::span<const float> raw_weight,
                          std::size_t output_width) {
  require(input.size() == rows * input_width,
          "CPU Qwen3.5 linear input shape mismatch");
  require(raw_weight.size() == output_width * input_width,
          "CPU Qwen3.5 linear weight shape mismatch");
  std::vector<float> output(rows * output_width, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < output_width; ++out) {
      double sum = 0.0;
      for (std::size_t in = 0; in < input_width; ++in) {
        sum += static_cast<double>(input[row * input_width + in]) *
               static_cast<double>(raw_weight[out * input_width + in]);
      }
      output[row * output_width + out] = static_cast<float>(sum);
    }
  }
  return output;
}

void residual_add(std::span<float> hidden, std::span<const float> residual) {
  require(hidden.size() == residual.size(),
          "CPU Qwen3.5 residual shape mismatch");
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    hidden[index] += residual[index];
  }
}

std::vector<float> transpose_output_weight(std::span<const float> raw_weight,
                                           std::size_t input_width,
                                           std::size_t output_width) {
  require(raw_weight.size() == input_width * output_width,
          "CPU Qwen3.5 output projection shape mismatch");
  std::vector<float> transposed(raw_weight.size());
  for (std::size_t input = 0; input < input_width; ++input) {
    for (std::size_t output = 0; output < output_width; ++output) {
      transposed[input * output_width + output] =
          raw_weight[output * input_width + input];
    }
  }
  return transposed;
}

std::vector<float>
run_linear_mixer(const model::Model &model,
                 const model::Qwen35LinearAttentionTensors &tensors,
                 const model::Qwen35Config &config,
                 std::span<const float> normalized, std::size_t tokens) {
  const std::size_t key_width =
      config.linear_key_head_count * config.linear_head_dimension;
  const std::size_t value_width =
      config.linear_value_head_count * config.linear_head_dimension;
  const std::size_t qkv_width = 2 * key_width + value_width;
  const std::size_t value_heads = config.linear_value_head_count;
  const std::size_t packed_width =
      qkv_width + value_heads + value_heads + value_width;

  const auto qkv = linear(normalized, tokens, config.hidden_size,
                          tensor_f32(model, *tensors.qkv), qkv_width);
  const auto beta = linear(normalized, tokens, config.hidden_size,
                           tensor_f32(model, *tensors.beta), value_heads);
  const auto alpha = linear(normalized, tokens, config.hidden_size,
                            tensor_f32(model, *tensors.alpha), value_heads);
  const auto gate = linear(normalized, tokens, config.hidden_size,
                           tensor_f32(model, *tensors.gate), value_width);

  std::vector<float> packed(tokens * packed_width);
  for (std::size_t token = 0; token < tokens; ++token) {
    auto destination =
        std::span<float>(packed).subspan(token * packed_width, packed_width);
    std::size_t offset = 0;
    const auto append = [&](std::span<const float> source) {
      std::copy(source.begin(), source.end(),
                destination.begin() + static_cast<std::ptrdiff_t>(offset));
      offset += source.size();
    };
    append(std::span<const float>(qkv).subspan(token * qkv_width, qkv_width));
    append(
        std::span<const float>(beta).subspan(token * value_heads, value_heads));
    append(std::span<const float>(alpha).subspan(token * value_heads,
                                                 value_heads));
    append(
        std::span<const float>(gate).subspan(token * value_width, value_width));
    require(offset == packed_width, "CPU Qwen3.5 linear packed width mismatch");
  }

  const auto args = GatedDeltaReferenceArgs::from_config(config, tokens);
  GatedDeltaReferenceState state(args);
  std::vector<float> delta_output(tokens * config.hidden_size);
  qwen35_gated_delta_prefill(
      packed, delta_output,
      GatedDeltaReferenceWeights{
          .conv_weight = tensor_f32(model, *tensors.convolution),
          .recurrent_a = tensor_f32(model, *tensors.recurrent_a),
          .dt_bias = tensor_f32(model, *tensors.time_step_bias),
          .output_norm_weight = tensor_f32(model, *tensors.output_norm),
      },
      args, state);
  return linear(delta_output, tokens, value_width,
                tensor_f32(model, *tensors.output), config.hidden_size);
}

std::vector<float>
run_full_mixer(const model::Model &model,
               const model::Qwen35FullAttentionTensors &tensors,
               const model::Qwen35Config &config,
               std::span<const float> normalized, std::size_t tokens) {
  const std::size_t query_width =
      config.full_attention_head_count * config.full_attention_head_dimension;
  const std::size_t kv_width = config.full_attention_kv_head_count *
                               config.full_attention_head_dimension;
  const auto query_gate =
      linear(normalized, tokens, config.hidden_size,
             tensor_f32(model, *tensors.query), 2 * query_width);
  const auto key = linear(normalized, tokens, config.hidden_size,
                          tensor_f32(model, *tensors.key), kv_width);
  const auto value = linear(normalized, tokens, config.hidden_size,
                            tensor_f32(model, *tensors.value), kv_width);

  const std::size_t packed_width =
      query_width + kv_width + kv_width + config.hidden_size;
  std::vector<float> packed(tokens * packed_width);
  for (std::size_t token = 0; token < tokens; ++token) {
    auto destination =
        std::span<float>(packed).subspan(token * packed_width, packed_width);
    const std::size_t source_base = token * 2 * query_width;
    for (std::size_t head = 0; head < config.full_attention_head_count;
         ++head) {
      for (std::size_t dim = 0; dim < config.full_attention_head_dimension;
           ++dim) {
        const std::size_t output_index =
            head * config.full_attention_head_dimension + dim;
        const std::size_t input_index =
            source_base + head * 2 * config.full_attention_head_dimension + dim;
        destination[output_index] = query_gate[input_index];
        destination[query_width + 2 * kv_width + output_index] =
            query_gate[input_index + config.full_attention_head_dimension];
      }
    }
    std::copy_n(key.begin() + static_cast<std::ptrdiff_t>(token * kv_width),
                kv_width,
                destination.begin() + static_cast<std::ptrdiff_t>(query_width));
    std::copy_n(value.begin() + static_cast<std::ptrdiff_t>(token * kv_width),
                kv_width,
                destination.begin() +
                    static_cast<std::ptrdiff_t>(query_width + kv_width));
  }

  const auto raw_output = tensor_f32(model, *tensors.output);
  const auto output_weight =
      transpose_output_weight(raw_output, query_width, config.hidden_size);
  std::vector<float> output(tokens * config.hidden_size);
  qwen35_gated_full_attention(
      packed, output,
      FullAttentionReferenceWeights{
          .query_norm_weight = tensor_f32(model, *tensors.query_norm),
          .key_norm_weight = tensor_f32(model, *tensors.key_norm),
          .output_weight = output_weight,
      },
      FullAttentionReferenceArgs::from_config(config, tokens, 0));
  return output;
}

} // namespace

Qwen35ReferenceExecution
qwen35_execute_model(const model::Model &model,
                     std::span<const std::int32_t> tokens) {
  const auto &config = model.qwen35_config();
  const auto &manifest = model.qwen35_manifest();
  require(!tokens.empty(), "CPU Qwen3.5 token span must not be empty");
  require(tokens.size() <= config.context_length,
          "CPU Qwen3.5 token span exceeds context length");
  for (const std::int32_t token : tokens) {
    require(token >= 0 &&
                static_cast<std::size_t>(token) < config.vocabulary_size,
            "CPU Qwen3.5 token id is out of range");
  }

  const std::size_t rows = tokens.size();
  const auto embedding = tensor_f32(model, *manifest.token_embedding);
  std::vector<float> hidden(rows * config.hidden_size);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t source =
        static_cast<std::size_t>(tokens[row]) * config.hidden_size;
    std::copy_n(embedding.begin() + static_cast<std::ptrdiff_t>(source),
                config.hidden_size,
                hidden.begin() +
                    static_cast<std::ptrdiff_t>(row * config.hidden_size));
  }

  for (const auto &layer : manifest.layers) {
    std::vector<float> normalized(hidden.size());
    qwen35_rms_norm(hidden, tensor_f32(model, *layer.common.input_norm),
                    normalized, rows, config.hidden_size,
                    config.rms_norm_epsilon);
    std::vector<float> mixed;
    if (layer.linear_attention.has_value()) {
      mixed = run_linear_mixer(model, *layer.linear_attention, config,
                               normalized, rows);
    } else {
      require(layer.full_attention.has_value(),
              "CPU Qwen3.5 layer has no attention branch");
      mixed = run_full_mixer(model, *layer.full_attention, config, normalized,
                             rows);
    }
    residual_add(hidden, mixed);

    qwen35_rms_norm(
        hidden, tensor_f32(model, *layer.common.post_attention_norm),
        normalized, rows, config.hidden_size, config.rms_norm_epsilon);
    auto gate = linear(normalized, rows, config.hidden_size,
                       tensor_f32(model, *layer.common.ffn_gate),
                       config.intermediate_size);
    const auto up = linear(normalized, rows, config.hidden_size,
                           tensor_f32(model, *layer.common.ffn_up),
                           config.intermediate_size);
    for (std::size_t index = 0; index < gate.size(); ++index) {
      gate[index] = gate[index] / (1.0F + std::exp(-gate[index])) * up[index];
    }
    const auto down =
        linear(gate, rows, config.intermediate_size,
               tensor_f32(model, *layer.common.ffn_down), config.hidden_size);
    residual_add(hidden, down);
  }

  const auto last_hidden = std::span<const float>(hidden).subspan(
      (rows - 1) * config.hidden_size, config.hidden_size);
  std::vector<float> final_hidden(config.hidden_size);
  qwen35_rms_norm(last_hidden, tensor_f32(model, *manifest.output_norm),
                  final_hidden, 1, config.hidden_size, config.rms_norm_epsilon);
  auto logits =
      linear(final_hidden, 1, config.hidden_size,
             tensor_f32(model, *manifest.output), config.vocabulary_size);
  const std::size_t token = argmax(logits);
  require(token <= static_cast<std::size_t>(
                       std::numeric_limits<std::int32_t>::max()),
          "CPU Qwen3.5 argmax exceeds int32 token range");
  return Qwen35ReferenceExecution{
      .logits = std::move(logits),
      .token = static_cast<std::int32_t>(token),
  };
}

} // namespace brt::reference
