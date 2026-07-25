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

void require(bool condition, const char* message) {
  if (!condition) {
    throw Qwen35DeltaError(message);
  }
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs,
                        const char* message) {
  require(rhs <= std::numeric_limits<std::size_t>::max() - lhs, message);
  return lhs + rhs;
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs,
                        const char* message) {
  require(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs,
          message);
  return lhs * rhs;
}

int checked_block_count(std::size_t blocks, const char* message) {
  require(blocks <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          message);
  return static_cast<int>(blocks);
}

void require_dtype(BrtDataType dtype) {
  require(dtype == BRT_DTYPE_F16 || dtype == BRT_DTYPE_BF16,
          "unsupported Qwen3.5 gated-delta dtype");
}

void require_aligned(const void* pointer, std::size_t alignment,
                     const char* message) {
  require(reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0, message);
}

CheckedShape validate_shape(const GatedDeltaShape& shape) {
  require(shape.tokens > 0, "gated-delta tokens must be positive");
  require(shape.hidden_size > 0,
          "gated-delta hidden_size must be positive");
  require(shape.key_heads > 0, "gated-delta key_heads must be positive");
  require(shape.value_heads > 0,
          "gated-delta value_heads must be positive");
  require(shape.key_dim > 0, "gated-delta key_dim must be positive");
  require(shape.value_dim > 0, "gated-delta value_dim must be positive");
  require(shape.conv_width > 0,
          "gated-delta conv_width must be positive");
  require(std::isfinite(shape.epsilon) && shape.epsilon >= 0.0F,
          "gated-delta epsilon must be finite and non-negative");
  require(shape.value_heads % shape.key_heads == 0,
          "gated-delta value_heads must be divisible by key_heads");

  const std::size_t key_size =
      checked_mul(shape.key_heads, shape.key_dim,
                  "gated-delta key shape overflow");
  const std::size_t value_size =
      checked_mul(shape.value_heads, shape.value_dim,
                  "gated-delta value shape overflow");
  require(value_size == shape.hidden_size,
          "gated-delta value_heads * value_dim must equal hidden_size");
  const std::size_t two_key_size =
      checked_mul(key_size, 2, "gated-delta convolution shape overflow");
  const std::size_t conv_dim =
      checked_add(two_key_size, value_size,
                  "gated-delta convolution shape overflow");
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
  const std::size_t recurrent_elements =
      checked_mul(value_size, shape.key_dim,
                  "gated-delta recurrent state shape overflow");
  const std::size_t workspace_elements =
      checked_add(conv_dim, shape.hidden_size,
                  "gated-delta workspace shape overflow");
  const std::size_t workspace_bytes =
      checked_mul(workspace_elements, sizeof(float),
                  "gated-delta workspace byte size overflow");

  (void)checked_block_count(shape.tokens,
                            "gated-delta token count exceeds launch limit");
  (void)checked_block_count(conv_dim,
                            "gated-delta convolution grid overflow");
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

void check_launch(const char* name) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw Qwen35DeltaError(std::string{name} + " launch failed: " +
                           cudaGetErrorString(error));
  }
}

template <typename T>
__device__ float load_as_float(const void* data, std::size_t index) {
  return static_cast<float>(static_cast<const T*>(data)[index]);
}

template <>
__device__ float load_as_float<__half>(const void* data, std::size_t index) {
  return __half2float(static_cast<const __half*>(data)[index]);
}

template <>
__device__ float load_as_float<__nv_bfloat16>(const void* data,
                                              std::size_t index) {
  return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
}

template <typename T>
__device__ void store_from_float(void* data, std::size_t index, float value) {
  static_cast<T*>(data)[index] = static_cast<T>(value);
}

template <>
__device__ void store_from_float<__half>(void* data, std::size_t index,
                                         float value) {
  static_cast<__half*>(data)[index] = __float2half_rn(value);
}

template <>
__device__ void store_from_float<__nv_bfloat16>(void* data, std::size_t index,
                                                float value) {
  static_cast<__nv_bfloat16*>(data)[index] = __float2bfloat16_rn(value);
}

__device__ float silu(float value) {
  return value / (1.0F + expf(-value));
}

__device__ float sigmoid(float value) {
  return 1.0F / (1.0F + expf(-value));
}

__device__ float softplus(float value) {
  if (value > 20.0F) {
    return value;
  }
  if (value < -20.0F) {
    return expf(value);
  }
  return log1pf(expf(value));
}

template <typename T>
__global__ void convolution_kernel(
    const void* input, const void* conv_weight, float* convolution_state,
    float* convolved, std::size_t input_base, std::size_t conv_dim,
    std::size_t conv_width) {
  const std::size_t channel =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= conv_dim) return;

  const float current = load_as_float<T>(input, input_base + channel);
  float sum =
      current *
      load_as_float<T>(conv_weight,
                       channel * conv_width + (conv_width - 1));
  if (conv_width > 1) {
    const std::size_t state_base = channel * (conv_width - 1);
    const std::size_t weight_base = channel * conv_width;
    for (std::size_t history = 0; history + 1 < conv_width; ++history) {
      sum += convolution_state[state_base + history] *
             load_as_float<T>(conv_weight, weight_base + history);
    }
    for (std::size_t history = 0; history + 2 < conv_width; ++history) {
      convolution_state[state_base + history] =
          convolution_state[state_base + history + 1];
    }
    convolution_state[state_base + conv_width - 2] = current;
  }
  convolved[channel] = silu(sum);
}

__global__ void qk_normalize_kernel(float* convolved, std::size_t key_heads,
                                    std::size_t key_dim) {
  extern __shared__ float shared[];
  const std::size_t vector = blockIdx.x;
  const bool query = vector < key_heads;
  const std::size_t head = query ? vector : vector - key_heads;
  const std::size_t base =
      (query ? 0 : key_heads * key_dim) + head * key_dim;

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

  const float scale =
      rsqrtf(shared[0] + kQkNormEpsilon) *
      (query ? rsqrtf(static_cast<float>(key_dim)) : 1.0F);
  for (std::size_t dim = threadIdx.x; dim < key_dim; dim += blockDim.x) {
    convolved[base + dim] *= scale;
  }
}

template <typename T>
__global__ void recurrent_kernel(
    const void* input, const void* a_log, const void* dt_bias,
    const float* convolved, float* recurrent_state, float* raw_output,
    std::size_t input_base, std::size_t conv_dim, std::size_t key_heads,
    std::size_t value_heads, std::size_t key_dim, std::size_t value_dim) {
  const std::size_t value_head = blockIdx.x;
  const std::size_t repeat = value_heads / key_heads;
  const std::size_t key_head = value_head / repeat;
  const std::size_t q_base = key_head * key_dim;
  const std::size_t k_base = key_heads * key_dim + key_head * key_dim;
  const std::size_t v_base =
      2 * key_heads * key_dim + value_head * value_dim;
  const float beta =
      sigmoid(load_as_float<T>(input, input_base + conv_dim + value_head));
  const float dt = softplus(
      load_as_float<T>(input,
                       input_base + conv_dim + value_heads + value_head) +
      load_as_float<T>(dt_bias, value_head));
  const float decay =
      expf(-expf(load_as_float<T>(a_log, value_head)) * dt);
  const std::size_t state_head_base = value_head * key_dim * value_dim;

  for (std::size_t value_index = threadIdx.x; value_index < value_dim;
       value_index += blockDim.x) {
    float memory_projection = 0.0F;
    for (std::size_t key_index = 0; key_index < key_dim; ++key_index) {
      const std::size_t state_index =
          state_head_base + key_index * value_dim + value_index;
      const float decayed = recurrent_state[state_index] * decay;
      recurrent_state[state_index] = decayed;
      memory_projection += decayed * convolved[k_base + key_index];
    }

    const float delta =
        (convolved[v_base + value_index] - memory_projection) * beta;
    float output_value = 0.0F;
    for (std::size_t key_index = 0; key_index < key_dim; ++key_index) {
      const std::size_t state_index =
          state_head_base + key_index * value_dim + value_index;
      const float updated =
          recurrent_state[state_index] +
          convolved[k_base + key_index] * delta;
      recurrent_state[state_index] = updated;
      output_value += updated * convolved[q_base + key_index];
    }
    raw_output[value_head * value_dim + value_index] = output_value;
  }
}

template <typename T>
__global__ void output_norm_gate_kernel(
    const void* input, const void* output_norm_weight,
    const float* raw_output, void* output, std::size_t input_base,
    std::size_t output_base, std::size_t conv_dim, std::size_t value_heads,
    std::size_t value_dim, float epsilon) {
  extern __shared__ float shared[];
  const std::size_t value_head = blockIdx.x;
  const std::size_t head_base = value_head * value_dim;

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
  const std::size_t gate_base =
      input_base + conv_dim + 2 * value_heads + head_base;
  for (std::size_t value_index = threadIdx.x; value_index < value_dim;
       value_index += blockDim.x) {
    const std::size_t index = head_base + value_index;
    const float normalized =
        raw_output[index] * scale *
        load_as_float<T>(output_norm_weight, value_index);
    const float gate = load_as_float<T>(input, gate_base + value_index);
    store_from_float<T>(output, output_base + index,
                        normalized * silu(gate));
  }
}

template <typename T>
void launch_typed(const void* input, const void* conv_weight,
                  const void* a_log, const void* dt_bias,
                  const void* output_norm_weight, void* output,
                  float* convolution_state, float* recurrent_state,
                  float* convolved, float* raw_output,
                  const GatedDeltaShape& shape, const CheckedShape& checked,
                  cudaStream_t stream) {
  const int convolution_grid = static_cast<int>(
      checked.conv_dim / static_cast<std::size_t>(kBlockSize) +
      (checked.conv_dim % static_cast<std::size_t>(kBlockSize) == 0 ? 0 : 1));
  const int qk_grid = static_cast<int>(2 * shape.key_heads);
  const int recurrent_grid = static_cast<int>(shape.value_heads);
  constexpr std::size_t shared_bytes = kBlockSize * sizeof(float);

  for (std::size_t token = 0; token < shape.tokens; ++token) {
    const std::size_t input_base = token * checked.token_stride;
    const std::size_t output_base = token * shape.hidden_size;
    convolution_kernel<T><<<convolution_grid, kBlockSize, 0, stream>>>(
        input, conv_weight, convolution_state, convolved, input_base,
        checked.conv_dim, shape.conv_width);
    check_launch("qwen35_gated_delta convolution");
    qk_normalize_kernel<<<qk_grid, kBlockSize, shared_bytes, stream>>>(
        convolved, shape.key_heads, shape.key_dim);
    check_launch("qwen35_gated_delta Q/K normalization");
    recurrent_kernel<T><<<recurrent_grid, kBlockSize, 0, stream>>>(
        input, a_log, dt_bias, convolved, recurrent_state, raw_output,
        input_base, checked.conv_dim, shape.key_heads, shape.value_heads,
        shape.key_dim, shape.value_dim);
    check_launch("qwen35_gated_delta recurrent update");
    output_norm_gate_kernel<T>
        <<<recurrent_grid, kBlockSize, shared_bytes, stream>>>(
            input, output_norm_weight, raw_output, output, input_base,
            output_base, checked.conv_dim, shape.value_heads, shape.value_dim,
            shape.epsilon);
    check_launch("qwen35_gated_delta output normalization");
  }
}

}  // namespace

std::size_t
qwen35_gated_delta_workspace_bytes(const GatedDeltaShape& shape) {
  return validate_shape(shape).workspace_bytes;
}

void qwen35_gated_delta(
    const void* input, const void* conv_weight, const void* a_log,
    const void* dt_bias, const void* output_norm_weight, void* output,
    float* convolution_state, float* recurrent_state, void* workspace,
    std::size_t workspace_bytes, GatedDeltaShape shape, BrtDataType dtype,
    cudaStream_t stream) {
  const CheckedShape checked = validate_shape(shape);
  require_dtype(dtype);
  require(input != nullptr, "gated-delta input pointer is null");
  require(conv_weight != nullptr,
          "gated-delta convolution weight pointer is null");
  require(a_log != nullptr, "gated-delta A-log pointer is null");
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
      dtype == BRT_DTYPE_F16 ? alignof(__half) : alignof(__nv_bfloat16);
  require_aligned(input, dtype_alignment,
                  "gated-delta input is not dtype-aligned");
  require_aligned(conv_weight, dtype_alignment,
                  "gated-delta convolution weight is not dtype-aligned");
  require_aligned(a_log, dtype_alignment,
                  "gated-delta A-log weight is not dtype-aligned");
  require_aligned(dt_bias, dtype_alignment,
                  "gated-delta dt-bias weight is not dtype-aligned");
  require_aligned(output_norm_weight, dtype_alignment,
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

  auto* convolved = static_cast<float*>(workspace);
  auto* raw_output = convolved + checked.conv_dim;
  if (dtype == BRT_DTYPE_F16) {
    launch_typed<__half>(
        input, conv_weight, a_log, dt_bias, output_norm_weight, output,
        convolution_state, recurrent_state, convolved, raw_output, shape,
        checked, stream);
    return;
  }
  launch_typed<__nv_bfloat16>(
      input, conv_weight, a_log, dt_bias, output_norm_weight, output,
      convolution_state, recurrent_state, convolved, raw_output, shape, checked,
      stream);
}

}  // namespace brt::kernels
