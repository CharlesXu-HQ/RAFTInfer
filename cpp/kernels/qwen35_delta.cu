#include "qwen35_delta.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace brt::kernels {
namespace {

constexpr int kBlockSize = 256;
constexpr float kQkNormEpsilon = 1.0e-6F;

struct CheckedShape {
  std::size_t key_size;
  std::size_t value_size;
  std::size_t conv_dim;
  std::size_t token_stride;
  std::size_t convolution_elements;
  std::size_t recurrent_elements;
  std::size_t workspace_bytes;
};

void require(bool condition, const char *message) {
  if (!condition) {
    throw Qwen35DeltaError(message);
  }
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char *message) {
  require(rhs <= std::numeric_limits<std::size_t>::max() - lhs, message);
  return lhs + rhs;
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, const char *message) {
  require(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs,
          message);
  return lhs * rhs;
}

int checked_block_count(std::size_t blocks, const char *message) {
  require(blocks <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          message);
  return static_cast<int>(blocks);
}

void require_dtype(BrtDataType dtype) {
  require(dtype == BRT_DTYPE_F32 || dtype == BRT_DTYPE_F16 ||
              dtype == BRT_DTYPE_BF16,
          "unsupported Qwen3.5 gated-delta dtype");
}

void require_weight_dtype(BrtDataType dtype, BrtDataType weight_dtype) {
  require(weight_dtype == BRT_DTYPE_F32 || weight_dtype == dtype ||
              (dtype == BRT_DTYPE_F32 && (weight_dtype == BRT_DTYPE_F16 ||
                                          weight_dtype == BRT_DTYPE_BF16)),
          "Qwen3.5 gated-delta weight dtype is incompatible with activations");
}

void require_aligned(const void *pointer, std::size_t alignment,
                     const char *message) {
  require(reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0, message);
}

CheckedShape validate_shape(const GatedDeltaShape &shape) {
  require(shape.tokens > 0, "gated-delta tokens must be positive");
  require(shape.hidden_size > 0, "gated-delta hidden_size must be positive");
  require(shape.key_heads > 0, "gated-delta key_heads must be positive");
  require(shape.value_heads > 0, "gated-delta value_heads must be positive");
  require(shape.key_dim > 0, "gated-delta key_dim must be positive");
  require(shape.value_dim > 0, "gated-delta value_dim must be positive");
  require(shape.conv_width > 0, "gated-delta conv_width must be positive");
  require(std::isfinite(shape.epsilon) && shape.epsilon >= 0.0F,
          "gated-delta epsilon must be finite and non-negative");
  require(shape.value_heads % shape.key_heads == 0,
          "gated-delta value_heads must be divisible by key_heads");

  const std::size_t key_size = checked_mul(shape.key_heads, shape.key_dim,
                                           "gated-delta key shape overflow");
  const std::size_t value_size = checked_mul(
      shape.value_heads, shape.value_dim, "gated-delta value shape overflow");
  require(value_size == shape.hidden_size,
          "gated-delta value_heads * value_dim must equal hidden_size");
  const std::size_t two_key_size =
      checked_mul(key_size, 2, "gated-delta convolution shape overflow");
  const std::size_t conv_dim = checked_add(
      two_key_size, value_size, "gated-delta convolution shape overflow");
  const std::size_t two_value_heads =
      checked_mul(shape.value_heads, 2, "gated-delta input shape overflow");
  const std::size_t token_stride =
      checked_add(checked_add(conv_dim, two_value_heads,
                              "gated-delta input shape overflow"),
                  shape.hidden_size, "gated-delta input shape overflow");
  (void)checked_mul(shape.tokens, token_stride,
                    "gated-delta input shape overflow");
  (void)checked_mul(shape.tokens, shape.hidden_size,
                    "gated-delta output shape overflow");
  (void)checked_mul(conv_dim, shape.conv_width,
                    "gated-delta convolution weight shape overflow");
  const std::size_t convolution_elements =
      checked_mul(conv_dim, shape.conv_width - 1,
                  "gated-delta convolution state shape overflow");
  const std::size_t recurrent_elements = checked_mul(
      value_size, shape.key_dim, "gated-delta recurrent state shape overflow");
  const std::size_t workspace_elements =
      checked_mul(shape.tokens,
                  checked_add(conv_dim, shape.hidden_size,
                              "gated-delta workspace shape overflow"),
                  "gated-delta workspace shape overflow");
  const std::size_t workspace_bytes =
      checked_mul(workspace_elements, sizeof(float),
                  "gated-delta workspace byte size overflow");

  (void)checked_block_count(shape.tokens,
                            "gated-delta token count exceeds launch limit");
  (void)checked_block_count(conv_dim, "gated-delta convolution grid overflow");
  (void)checked_block_count(
      checked_mul(shape.key_heads, 2,
                  "gated-delta Q/K normalization grid overflow"),
      "gated-delta Q/K normalization grid overflow");
  (void)checked_block_count(shape.value_heads,
                            "gated-delta recurrent grid overflow");

  return CheckedShape{
      .key_size = key_size,
      .value_size = value_size,
      .conv_dim = conv_dim,
      .token_stride = token_stride,
      .convolution_elements = convolution_elements,
      .recurrent_elements = recurrent_elements,
      .workspace_bytes = workspace_bytes,
  };
}

void check_launch(const char *name) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw Qwen35DeltaError(std::string{name} +
                           " launch failed: " + cudaGetErrorString(error));
  }
}

template <typename T>
__device__ float load_as_float(const void *data, std::size_t index) {
  return static_cast<float>(static_cast<const T *>(data)[index]);
}

template <>
__device__ float load_as_float<__half>(const void *data, std::size_t index) {
  return __half2float(static_cast<const __half *>(data)[index]);
}

template <>
__device__ float load_as_float<__nv_bfloat16>(const void *data,
                                              std::size_t index) {
  return __bfloat162float(static_cast<const __nv_bfloat16 *>(data)[index]);
}

template <typename T>
__device__ void store_from_float(void *data, std::size_t index, float value) {
  static_cast<T *>(data)[index] = static_cast<T>(value);
}

template <>
__device__ void store_from_float<__half>(void *data, std::size_t index,
                                         float value) {
  static_cast<__half *>(data)[index] = __float2half_rn(value);
}

template <>
__device__ void store_from_float<__nv_bfloat16>(void *data, std::size_t index,
                                                float value) {
  static_cast<__nv_bfloat16 *>(data)[index] = __float2bfloat16_rn(value);
}

__device__ float silu(float value) { return value / (1.0F + expf(-value)); }

__device__ float sigmoid(float value) { return 1.0F / (1.0F + expf(-value)); }

__device__ float softplus(float value) {
  if (value > 20.0F) {
    return value;
  }
  if (value < -20.0F) {
    return expf(value);
  }
  return log1pf(expf(value));
}

template <typename T, typename Weight>
__global__ void
batched_convolution_kernel(const void *input, const void *conv_weight,
                           float *convolution_state, float *convolved,
                           std::size_t tokens, std::size_t token_stride,
                           std::size_t conv_dim, std::size_t conv_width) {
  const std::size_t channel =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= conv_dim)
    return;

  const std::size_t history_count = conv_width - 1;
  const std::size_t state_base = channel * history_count;
  const std::size_t weight_base = channel * conv_width;
  for (std::size_t token = 0; token < tokens; ++token) {
    const float current =
        load_as_float<T>(input, token * token_stride + channel);
    float sum = current *
                load_as_float<Weight>(conv_weight, weight_base + history_count);
    for (std::size_t history = 0; history < history_count; ++history) {
      const std::size_t source_offset = token + history;
      const float previous =
          source_offset < history_count
              ? convolution_state[state_base + source_offset]
              : load_as_float<T>(
                    input,
                    (source_offset - history_count) * token_stride + channel);
      sum +=
          previous * load_as_float<Weight>(conv_weight, weight_base + history);
    }
    convolved[token * conv_dim + channel] = silu(sum);
  }

  for (std::size_t history = 0; history < history_count; ++history) {
    const std::size_t source_offset = tokens + history;
    convolution_state[state_base + history] =
        source_offset < history_count
            ? convolution_state[state_base + source_offset]
            : load_as_float<T>(input,
                               (source_offset - history_count) * token_stride +
                                   channel);
  }
}

__global__ void batched_qk_normalize_kernel(float *convolved,
                                            std::size_t tokens,
                                            std::size_t conv_dim,
                                            std::size_t key_heads,
                                            std::size_t key_dim) {
  extern __shared__ float shared[];
  const std::size_t vectors_per_token = 2 * key_heads;
  const std::size_t flat_vector = blockIdx.x;
  const std::size_t token = flat_vector / vectors_per_token;
  if (token >= tokens)
    return;
  const std::size_t vector = flat_vector % vectors_per_token;
  const bool query = vector < key_heads;
  const std::size_t head = query ? vector : vector - key_heads;
  const std::size_t base =
      token * conv_dim + (query ? 0 : key_heads * key_dim) + head * key_dim;

  double thread_sum = 0.0;
  for (std::size_t dim = threadIdx.x; dim < key_dim; dim += blockDim.x) {
    const float value = convolved[base + dim];
    thread_sum += static_cast<double>(value) * static_cast<double>(value);
  }
  shared[threadIdx.x] = static_cast<float>(thread_sum);
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float scale = rsqrtf(shared[0] + kQkNormEpsilon) *
                      (query ? rsqrtf(static_cast<float>(key_dim)) : 1.0F);
  for (std::size_t dim = threadIdx.x; dim < key_dim; dim += blockDim.x) {
    convolved[base + dim] *= scale;
  }
}

template <typename T, typename Weight>
__global__ void tiled_recurrent_kernel(
    const void *input, const void *recurrent_a, const void *dt_bias,
    const float *convolved, float *recurrent_state, float *raw_output,
    std::size_t tokens, std::size_t token_stride, std::size_t conv_dim,
    std::size_t key_heads, std::size_t value_heads, std::size_t key_dim,
    std::size_t value_dim, std::size_t value_tiles) {
  constexpr std::size_t kValueTile = 16;
  const std::size_t flat_block = blockIdx.x;
  const std::size_t value_head = flat_block / value_tiles;
  const std::size_t tile = flat_block % value_tiles;
  const std::size_t value_index =
      tile * kValueTile + static_cast<std::size_t>(threadIdx.x);
  if (value_head >= value_heads || value_index >= value_dim)
    return;

  const std::size_t key_head = value_head % key_heads;
  const std::size_t q_offset = key_head * key_dim;
  const std::size_t k_offset = key_heads * key_dim + key_head * key_dim;
  const std::size_t v_offset = 2 * key_heads * key_dim + value_head * value_dim;
  const std::size_t value_size = value_heads * value_dim;
  const std::size_t state_head_base = value_head * key_dim * value_dim;

  for (std::size_t token = 0; token < tokens; ++token) {
    const std::size_t input_base = token * token_stride;
    const std::size_t convolved_base = token * conv_dim;
    float beta = 0.0F;
    float decay = 0.0F;
    if (threadIdx.x == 0) {
      beta =
          sigmoid(load_as_float<T>(input, input_base + conv_dim + value_head));
      const float dt =
          softplus(load_as_float<T>(input, input_base + conv_dim + value_heads +
                                               value_head) +
                   load_as_float<Weight>(dt_bias, value_head));
      decay = expf(load_as_float<Weight>(recurrent_a, value_head) * dt);
    }
    const unsigned int active = __activemask();
    beta = __shfl_sync(active, beta, 0);
    decay = __shfl_sync(active, decay, 0);

    float memory_projection = 0.0F;
    for (std::size_t key_index = 0; key_index < key_dim; ++key_index) {
      const std::size_t state_index =
          state_head_base + key_index * value_dim + value_index;
      const float decayed = recurrent_state[state_index] * decay;
      memory_projection +=
          decayed * convolved[convolved_base + k_offset + key_index];
    }
    const float delta = (convolved[convolved_base + v_offset + value_index] -
                         memory_projection) *
                        beta;
    float output_value = 0.0F;
    for (std::size_t key_index = 0; key_index < key_dim; ++key_index) {
      const std::size_t state_index =
          state_head_base + key_index * value_dim + value_index;
      const float updated =
          recurrent_state[state_index] * decay +
          convolved[convolved_base + k_offset + key_index] * delta;
      recurrent_state[state_index] = updated;
      output_value +=
          updated * convolved[convolved_base + q_offset + key_index];
    }
    raw_output[token * value_size + value_head * value_dim + value_index] =
        output_value;
  }
}

template <typename T, typename Weight>
__global__ void register_recurrent_128_kernel(
    const void *input, const void *recurrent_a, const void *dt_bias,
    const float *convolved, float *recurrent_state, float *raw_output,
    std::size_t tokens, std::size_t token_stride, std::size_t conv_dim,
    std::size_t key_heads, std::size_t value_heads, std::size_t value_dim) {
  // Qwen3.5's 128-row state is kept in per-lane registers across the token
  // loop. This project-native specialization follows the register-resident
  // column strategy validated by the pinned llama.cpp reference.
  constexpr int kWarpSize = 32;
  constexpr int kRowsPerLane = 4;
  constexpr int kColumnsPerBlock = 4;
  const std::size_t value_head = blockIdx.x;
  const std::size_t value_index =
      static_cast<std::size_t>(blockIdx.y) * kColumnsPerBlock + threadIdx.y;
  if (value_head >= value_heads || value_index >= value_dim)
    return;

  const int lane = threadIdx.x;
  const std::size_t key_head = value_head % key_heads;
  constexpr std::size_t kKeyDim = kWarpSize * kRowsPerLane;
  const std::size_t q_offset = key_head * kKeyDim;
  const std::size_t k_offset = key_heads * kKeyDim + key_head * kKeyDim;
  const std::size_t v_offset = 2 * key_heads * kKeyDim + value_head * value_dim;
  const std::size_t value_size = value_heads * value_dim;
  const std::size_t state_head_base = value_head * kKeyDim * value_dim;

  float state_shard[kRowsPerLane];
#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const std::size_t key_index =
        static_cast<std::size_t>(row * kWarpSize + lane);
    state_shard[row] =
        recurrent_state[state_head_base + key_index * value_dim + value_index];
  }

  for (std::size_t token = 0; token < tokens; ++token) {
    const std::size_t input_base = token * token_stride;
    const std::size_t convolved_base = token * conv_dim;
    float beta = 0.0F;
    float decay = 0.0F;
    if (lane == 0) {
      beta =
          sigmoid(load_as_float<T>(input, input_base + conv_dim + value_head));
      const float dt =
          softplus(load_as_float<T>(input, input_base + conv_dim + value_heads +
                                               value_head) +
                   load_as_float<Weight>(dt_bias, value_head));
      decay = expf(load_as_float<Weight>(recurrent_a, value_head) * dt);
    }
    beta = __shfl_sync(0xFFFFFFFFU, beta, 0);
    decay = __shfl_sync(0xFFFFFFFFU, decay, 0);

    float q_shard[kRowsPerLane];
    float k_shard[kRowsPerLane];
    float memory_projection = 0.0F;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      const std::size_t key_index =
          static_cast<std::size_t>(row * kWarpSize + lane);
      q_shard[row] = convolved[convolved_base + q_offset + key_index];
      k_shard[row] = convolved[convolved_base + k_offset + key_index];
      memory_projection += state_shard[row] * decay * k_shard[row];
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      memory_projection +=
          __shfl_down_sync(0xFFFFFFFFU, memory_projection, offset);
    }
    memory_projection = __shfl_sync(0xFFFFFFFFU, memory_projection, 0);

    const float delta = (convolved[convolved_base + v_offset + value_index] -
                         memory_projection) *
                        beta;
    float output_value = 0.0F;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      state_shard[row] = state_shard[row] * decay + k_shard[row] * delta;
      output_value += state_shard[row] * q_shard[row];
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      output_value += __shfl_down_sync(0xFFFFFFFFU, output_value, offset);
    }
    if (lane == 0) {
      raw_output[token * value_size + value_head * value_dim + value_index] =
          output_value;
    }
  }

#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const std::size_t key_index =
        static_cast<std::size_t>(row * kWarpSize + lane);
    recurrent_state[state_head_base + key_index * value_dim + value_index] =
        state_shard[row];
  }
}

template <typename T, typename Weight>
__global__ void batched_output_norm_gate_kernel(
    const void *input, const void *output_norm_weight, const float *raw_output,
    void *output, std::size_t tokens, std::size_t token_stride,
    std::size_t conv_dim, std::size_t value_heads, std::size_t value_dim,
    float epsilon) {
  extern __shared__ float shared[];
  const std::size_t flat_head = blockIdx.x;
  const std::size_t token = flat_head / value_heads;
  if (token >= tokens)
    return;
  const std::size_t value_head = flat_head % value_heads;
  const std::size_t value_size = value_heads * value_dim;
  const std::size_t head_base = token * value_size + value_head * value_dim;

  double thread_sum = 0.0;
  for (std::size_t value_index = threadIdx.x; value_index < value_dim;
       value_index += blockDim.x) {
    const float value = raw_output[head_base + value_index];
    thread_sum += static_cast<double>(value) * static_cast<double>(value);
  }
  shared[threadIdx.x] = static_cast<float>(thread_sum);
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float scale =
      rsqrtf(shared[0] / static_cast<float>(value_dim) + epsilon);
  const std::size_t gate_base = token * token_stride + conv_dim +
                                2 * value_heads + value_head * value_dim;
  for (std::size_t value_index = threadIdx.x; value_index < value_dim;
       value_index += blockDim.x) {
    const float normalized =
        raw_output[head_base + value_index] * scale *
        load_as_float<Weight>(output_norm_weight, value_index);
    const float gate = load_as_float<T>(input, gate_base + value_index);
    store_from_float<T>(output, head_base + value_index,
                        normalized * silu(gate));
  }
}

template <typename T, typename Weight>
void launch_typed(const void *input, const void *conv_weight,
                  const void *recurrent_a, const void *dt_bias,
                  const void *output_norm_weight, void *output,
                  float *convolution_state, float *recurrent_state,
                  float *convolved, float *raw_output,
                  const GatedDeltaShape &shape, const CheckedShape &checked,
                  cudaStream_t stream) {
  constexpr std::size_t kValueTile = 16;
  const int convolution_grid = checked_block_count(
      checked.conv_dim / static_cast<std::size_t>(kBlockSize) +
          (checked.conv_dim % static_cast<std::size_t>(kBlockSize) == 0 ? 0
                                                                        : 1),
      "gated-delta convolution grid overflow");
  const int qk_grid = checked_block_count(
      checked_mul(shape.tokens,
                  checked_mul(shape.key_heads, std::size_t{2},
                              "gated-delta Q/K grid overflow"),
                  "gated-delta Q/K grid overflow"),
      "gated-delta Q/K grid overflow");
  const std::size_t value_tiles = shape.value_dim / kValueTile +
                                  (shape.value_dim % kValueTile == 0 ? 0 : 1);
  const int recurrent_grid =
      checked_block_count(checked_mul(shape.value_heads, value_tiles,
                                      "gated-delta recurrent grid overflow"),
                          "gated-delta recurrent grid overflow");
  const int output_grid =
      checked_block_count(checked_mul(shape.tokens, shape.value_heads,
                                      "gated-delta output grid overflow"),
                          "gated-delta output grid overflow");
  constexpr std::size_t shared_bytes = kBlockSize * sizeof(float);
  batched_convolution_kernel<T, Weight>
      <<<convolution_grid, kBlockSize, 0, stream>>>(
          input, conv_weight, convolution_state, convolved, shape.tokens,
          checked.token_stride, checked.conv_dim, shape.conv_width);
  check_launch("qwen35_gated_delta batched convolution");
  batched_qk_normalize_kernel<<<qk_grid, kBlockSize, shared_bytes, stream>>>(
      convolved, shape.tokens, checked.conv_dim, shape.key_heads,
      shape.key_dim);
  check_launch("qwen35_gated_delta batched Q/K normalization");
  if (shape.key_dim == 128) {
    constexpr int kWarpSize = 32;
    constexpr int kColumnsPerBlock = 4;
    const int register_column_blocks = checked_block_count(
        shape.value_dim / kColumnsPerBlock +
            (shape.value_dim % kColumnsPerBlock == 0 ? 0 : 1),
        "gated-delta register recurrent grid overflow");
    const dim3 register_grid{static_cast<unsigned int>(shape.value_heads),
                             static_cast<unsigned int>(register_column_blocks),
                             1};
    const dim3 register_block{kWarpSize, kColumnsPerBlock, 1};
    register_recurrent_128_kernel<T, Weight>
        <<<register_grid, register_block, 0, stream>>>(
            input, recurrent_a, dt_bias, convolved, recurrent_state, raw_output,
            shape.tokens, checked.token_stride, checked.conv_dim,
            shape.key_heads, shape.value_heads, shape.value_dim);
    check_launch("qwen35_gated_delta register recurrent update");
  } else {
    tiled_recurrent_kernel<T, Weight>
        <<<recurrent_grid, static_cast<int>(kValueTile), 0, stream>>>(
            input, recurrent_a, dt_bias, convolved, recurrent_state, raw_output,
            shape.tokens, checked.token_stride, checked.conv_dim,
            shape.key_heads, shape.value_heads, shape.key_dim, shape.value_dim,
            value_tiles);
    check_launch("qwen35_gated_delta tiled recurrent update");
  }
  batched_output_norm_gate_kernel<T, Weight>
      <<<output_grid, kBlockSize, shared_bytes, stream>>>(
          input, output_norm_weight, raw_output, output, shape.tokens,
          checked.token_stride, checked.conv_dim, shape.value_heads,
          shape.value_dim, shape.epsilon);
  check_launch("qwen35_gated_delta batched output normalization");
}

} // namespace

std::size_t qwen35_gated_delta_workspace_bytes(const GatedDeltaShape &shape) {
  return validate_shape(shape).workspace_bytes;
}

void qwen35_gated_delta(const void *input, const void *conv_weight,
                        const void *recurrent_a, const void *dt_bias,
                        const void *output_norm_weight, void *output,
                        float *convolution_state, float *recurrent_state,
                        void *workspace, std::size_t workspace_bytes,
                        GatedDeltaShape shape, BrtDataType dtype,
                        BrtDataType weight_dtype, cudaStream_t stream) {
  const CheckedShape checked = validate_shape(shape);
  require_dtype(dtype);
  require_weight_dtype(dtype, weight_dtype);
  require(input != nullptr, "gated-delta input pointer is null");
  require(conv_weight != nullptr,
          "gated-delta convolution weight pointer is null");
  require(recurrent_a != nullptr,
          "gated-delta recurrent coefficient pointer is null");
  require(dt_bias != nullptr, "gated-delta dt-bias pointer is null");
  require(output_norm_weight != nullptr,
          "gated-delta output norm weight pointer is null");
  require(output != nullptr, "gated-delta output pointer is null");
  require(checked.convolution_elements == 0 || convolution_state != nullptr,
          "gated-delta convolution state pointer is null");
  require(checked.recurrent_elements == 0 || recurrent_state != nullptr,
          "gated-delta recurrent state pointer is null");
  require(workspace != nullptr, "gated-delta workspace pointer is null");
  require(workspace_bytes >= checked.workspace_bytes,
          "gated-delta workspace is too small");
  const std::size_t dtype_alignment =
      dtype == BRT_DTYPE_F32
          ? alignof(float)
          : (dtype == BRT_DTYPE_F16 ? alignof(__half) : alignof(__nv_bfloat16));
  const std::size_t weight_alignment =
      weight_dtype == BRT_DTYPE_F32
          ? alignof(float)
          : (weight_dtype == BRT_DTYPE_F16 ? alignof(__half)
                                           : alignof(__nv_bfloat16));
  require_aligned(input, dtype_alignment,
                  "gated-delta input is not dtype-aligned");
  require_aligned(conv_weight, weight_alignment,
                  "gated-delta convolution weight is not dtype-aligned");
  require_aligned(recurrent_a, weight_alignment,
                  "gated-delta recurrent coefficient is not dtype-aligned");
  require_aligned(dt_bias, weight_alignment,
                  "gated-delta dt-bias weight is not dtype-aligned");
  require_aligned(output_norm_weight, weight_alignment,
                  "gated-delta output norm weight is not dtype-aligned");
  require_aligned(output, dtype_alignment,
                  "gated-delta output is not dtype-aligned");
  require_aligned(workspace, alignof(float),
                  "gated-delta workspace is not float-aligned");
  require_aligned(recurrent_state, alignof(float),
                  "gated-delta recurrent state is not float-aligned");
  if (checked.convolution_elements != 0) {
    require_aligned(convolution_state, alignof(float),
                    "gated-delta convolution state is not float-aligned");
  }
  require(stream != nullptr, "CUDA stream is null");

  auto *convolved = static_cast<float *>(workspace);
  auto *raw_output = convolved + shape.tokens * checked.conv_dim;
  const auto launch_for_weights = [&]<typename T>() {
    if (weight_dtype == BRT_DTYPE_F32) {
      launch_typed<T, float>(input, conv_weight, recurrent_a, dt_bias,
                             output_norm_weight, output, convolution_state,
                             recurrent_state, convolved, raw_output, shape,
                             checked, stream);
    } else if (weight_dtype == BRT_DTYPE_F16) {
      launch_typed<T, __half>(input, conv_weight, recurrent_a, dt_bias,
                              output_norm_weight, output, convolution_state,
                              recurrent_state, convolved, raw_output, shape,
                              checked, stream);
    } else {
      launch_typed<T, __nv_bfloat16>(
          input, conv_weight, recurrent_a, dt_bias, output_norm_weight, output,
          convolution_state, recurrent_state, convolved, raw_output, shape,
          checked, stream);
    }
  };
  if (dtype == BRT_DTYPE_F32) {
    launch_for_weights.template operator()<float>();
  } else if (dtype == BRT_DTYPE_F16) {
    launch_for_weights.template operator()<__half>();
  } else {
    launch_for_weights.template operator()<__nv_bfloat16>();
  }
}

} // namespace brt::kernels
