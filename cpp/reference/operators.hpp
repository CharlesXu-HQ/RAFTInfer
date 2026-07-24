#pragma once

#include "bf16.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace brt::reference {

struct AddShape {
  std::size_t elements;
};

struct EmbeddingShape {
  std::size_t tokens;
  std::size_t embedding_dim;
  std::size_t vocab_size;
};

struct LinearShape {
  std::size_t batch;
  std::size_t in_features;
  std::size_t out_features;
};

struct RmsNormShape {
  std::size_t rows;
  std::size_t cols;
};

struct RopeShape {
  std::size_t tokens;
  std::size_t heads;
  std::size_t head_dim;
  std::size_t rotary_dim;
  std::size_t position;
  float base;
  bool interleaved;
};

struct SoftmaxShape {
  std::size_t rows;
  std::size_t cols;
};

struct SwiGluShape {
  std::size_t elements;
};

void add(
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::span<float> output,
    AddShape shape);

std::size_t argmax(std::span<const float> values);

void bf16_linear(
    std::span<const bf16_t> input,
    std::span<const bf16_t> weight,
    std::span<float> output,
    LinearShape shape);

void embedding(
    std::span<const std::int32_t> tokens,
    std::span<const float> table,
    std::span<float> output,
    EmbeddingShape shape);

void rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    RmsNormShape shape,
    float epsilon);

void rope(
    std::span<const float> input,
    std::span<float> output,
    RopeShape shape);

void softmax(
    std::span<const float> input,
    std::span<float> output,
    SoftmaxShape shape);

void swiglu(
    std::span<const float> gate,
    std::span<const float> up,
    std::span<float> output,
    SwiGluShape shape);

}  // namespace brt::reference
