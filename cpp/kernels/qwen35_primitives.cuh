#pragma once

#include <brt/tensor.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace brt::kernels {

class Qwen35PrimitiveError : public std::runtime_error {
 public:
  explicit Qwen35PrimitiveError(const std::string& message)
      : std::runtime_error(message) {}
};

struct EmbeddingShape {
  std::size_t tokens;
  std::size_t embedding_dim;
  std::size_t vocab_size;
};

struct RmsNormShape {
  std::size_t rows;
  std::size_t cols;
};

struct QkNormRopeShape {
  std::size_t tokens;
  std::size_t heads;
  std::size_t head_dim;
  std::size_t rotary_dim;
  std::size_t position_offset;
  float rope_base;
};

void qwen35_validate_token_ids(std::span<const std::int32_t> tokens,
                               std::size_t vocab_size);

// `tokens` must be validated with `qwen35_validate_token_ids` before uploading
// or otherwise passing device-resident ids to this wrapper.
void qwen35_embedding(const std::int32_t* tokens, const void* table,
                      void* output, EmbeddingShape shape,
                      BrtDataType dtype, cudaStream_t stream);

void qwen35_rms_norm(const void* input, const void* weight, void* output,
                     RmsNormShape shape, float epsilon,
                     BrtDataType dtype, cudaStream_t stream);

void qwen35_residual_add(const void* lhs, const void* rhs, void* output,
                         std::size_t elements, BrtDataType dtype,
                         cudaStream_t stream);

void qwen35_qk_norm_rope(const void* input, const void* weight, void* output,
                         QkNormRopeShape shape, float epsilon,
                         BrtDataType dtype, cudaStream_t stream);

void qwen35_sigmoid_gate(const void* values, const void* gates, void* output,
                         std::size_t elements, BrtDataType dtype,
                         cudaStream_t stream);

void qwen35_swiglu(const void* gate, const void* up, void* output,
                   std::size_t elements, BrtDataType dtype,
                   cudaStream_t stream);

void qwen35_argmax(const float* logits, std::int32_t* output_index,
                   std::size_t elements, cudaStream_t stream);

void qwen35_argmax_typed(const void* logits, std::int32_t* output_index,
                         std::size_t elements, BrtDataType dtype,
                         cudaStream_t stream);

void qwen35_split_full_query_gate(const void* query_gate, void* query,
                                  void* gate, std::size_t tokens,
                                  std::size_t heads, std::size_t head_dim,
                                  BrtDataType dtype, cudaStream_t stream);

void qwen35_pack_linear_delta_input(
    const void* qkv, const void* beta, const void* alpha, const void* gate,
    void* packed, std::size_t tokens, std::size_t qkv_width,
    std::size_t beta_width, std::size_t alpha_width, std::size_t gate_width,
    BrtDataType dtype, cudaStream_t stream);

}  // namespace brt::kernels
