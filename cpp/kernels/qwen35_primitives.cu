#include "qwen35_primitives.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <limits>

namespace brt::kernels {
namespace {

constexpr int kBlockSize = 256;

void require(bool condition, const char* message) {
  if (!condition) {
    throw Qwen35PrimitiveError(message);
  }
}

void require_dtype(BrtDataType dtype) {
  require(dtype == BRT_DTYPE_F16 || dtype == BRT_DTYPE_BF16,
          "unsupported Qwen3.5 primitive dtype");
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs,
                        const char* message) {
  require(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs,
          message);
  return lhs * rhs;
}

void check_launch(const char* name) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw Qwen35PrimitiveError(std::string{name} + " launch failed: " +
                              cudaGetErrorString(error));
  }
}

template <typename T>
__device__ float load_as_float(const void* data, std::size_t index) {
  return static_cast<float>(static_cast<const T*>(data)[index]);
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

template <typename T>
__global__ void embedding_kernel(const std::int32_t* tokens, const void* table,
                                 void* output, std::size_t token_count,
                                 std::size_t embedding_dim,
                                 std::size_t vocab_size) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = token_count * embedding_dim;
  if (element >= total) return;
  const std::size_t token_index = element / embedding_dim;
  const std::size_t col = element - token_index * embedding_dim;
  const std::int32_t token = tokens[token_index];
  if (token < 0 || static_cast<std::size_t>(token) >= vocab_size) {
    store_from_float<T>(output, element, 0.0F);
    return;
  }
  const std::size_t table_index =
      static_cast<std::size_t>(token) * embedding_dim + col;
  store_from_float<T>(output, element, load_as_float<T>(table, table_index));
}

template <typename T>
__global__ void rms_norm_kernel(const void* input, const void* weight,
                                void* output, std::size_t cols,
                                float epsilon) {
  extern __shared__ float shared[];
  const std::size_t row = blockIdx.x;
  double thread_sum = 0.0;
  for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = load_as_float<T>(input, row * cols + col);
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
      rsqrtf(shared[0] / static_cast<float>(cols) + epsilon);
  for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const std::size_t index = row * cols + col;
    const float value = load_as_float<T>(input, index);
    const float weight_value = load_as_float<T>(weight, col);
    store_from_float<T>(output, index, value * scale * (1.0F + weight_value));
  }
}

template <typename T>
__global__ void add_kernel(const void* lhs, const void* rhs, void* output,
                           std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float sum = load_as_float<T>(lhs, index) + load_as_float<T>(rhs, index);
  store_from_float<T>(output, index, sum);
}

template <typename T>
__global__ void qk_norm_rope_kernel(const void* input, const void* weight,
                                    void* output, std::size_t heads,
                                    std::size_t head_dim,
                                    std::size_t rotary_dim,
                                    std::size_t position_offset,
                                    float rope_base, float epsilon) {
  extern __shared__ float shared[];
  const std::size_t vector = blockIdx.x;
  const std::size_t token = vector / heads;
  const std::size_t base = vector * head_dim;
  double thread_sum = 0.0;
  for (std::size_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float value = load_as_float<T>(input, base + dim);
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
      rsqrtf(shared[0] / static_cast<float>(head_dim) + epsilon);
  const std::size_t pair_count = rotary_dim / 2;
  for (std::size_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float value = load_as_float<T>(input, base + dim) * scale *
                  (1.0F + load_as_float<T>(weight, dim));
    if (dim < rotary_dim) {
      const bool first_half = dim < pair_count;
      const std::size_t pair = first_half ? dim : dim - pair_count;
      const std::size_t partner_dim = first_half ? dim + pair_count : pair;
      const float partner = load_as_float<T>(input, base + partner_dim) *
                            scale *
                            (1.0F + load_as_float<T>(weight, partner_dim));
      const double exponent =
          static_cast<double>(2 * pair) / static_cast<double>(rotary_dim);
      const float theta = static_cast<float>(
          static_cast<double>(position_offset + token) /
          pow(static_cast<double>(rope_base), exponent));
      float sin_theta = 0.0F;
      float cos_theta = 0.0F;
      sincosf(theta, &sin_theta, &cos_theta);
      value = first_half ? value * cos_theta - partner * sin_theta
                         : partner * sin_theta + value * cos_theta;
    }
    store_from_float<T>(output, base + dim, value);
  }
}

template <typename T>
__global__ void sigmoid_gate_kernel(const void* values, const void* gates,
                                    void* output, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = load_as_float<T>(values, index);
  const float gate = load_as_float<T>(gates, index);
  store_from_float<T>(output, index, value / (1.0F + expf(-gate)));
}

template <typename T>
__global__ void swiglu_kernel(const void* gate, const void* up, void* output,
                              std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float gated = load_as_float<T>(gate, index);
  const float up_value = load_as_float<T>(up, index);
  store_from_float<T>(output, index, silu(gated) * up_value);
}

__global__ void argmax_kernel(const float* logits, std::int32_t* output_index,
                              std::size_t elements) {
  extern __shared__ unsigned char raw_shared[];
  auto* shared_values = reinterpret_cast<float*>(raw_shared);
  auto* shared_indices =
      reinterpret_cast<std::size_t*>(shared_values + blockDim.x);

  float best_value = -std::numeric_limits<float>::infinity();
  std::size_t best_index = 0;
  for (std::size_t index = threadIdx.x; index < elements;
       index += blockDim.x) {
    const float value = logits[index];
    if (value > best_value ||
        (value == best_value && index < best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  shared_values[threadIdx.x] = best_value;
  shared_indices[threadIdx.x] = best_index;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_value = shared_values[threadIdx.x + stride];
      const std::size_t other_index = shared_indices[threadIdx.x + stride];
      if (other_value > shared_values[threadIdx.x] ||
          (other_value == shared_values[threadIdx.x] &&
           other_index < shared_indices[threadIdx.x])) {
        shared_values[threadIdx.x] = other_value;
        shared_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *output_index = static_cast<std::int32_t>(shared_indices[0]);
  }
}

int checked_grid_for(std::size_t elements, const char* message) {
  const auto block_size = static_cast<std::size_t>(kBlockSize);
  const std::size_t blocks =
      elements / block_size + (elements % block_size == 0 ? 0 : 1);
  require(blocks <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          message);
  return static_cast<int>(blocks);
}

int checked_block_count(std::size_t blocks, const char* message) {
  require(blocks <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          message);
  return static_cast<int>(blocks);
}

template <typename KernelLauncher>
void launch_by_dtype(BrtDataType dtype, KernelLauncher launcher) {
  if (dtype == BRT_DTYPE_F16) {
    launcher.template operator()<__half>();
    return;
  }
  if (dtype == BRT_DTYPE_BF16) {
    launcher.template operator()<__nv_bfloat16>();
    return;
  }
  require_dtype(dtype);
}

}  // namespace

void qwen35_embedding(const std::int32_t* tokens, const void* table,
                      void* output, EmbeddingShape shape,
                      BrtDataType dtype, cudaStream_t stream) {
  require(tokens != nullptr, "embedding tokens pointer is null");
  require(table != nullptr, "embedding table pointer is null");
  require(output != nullptr, "embedding output pointer is null");
  require(shape.tokens > 0, "embedding tokens must be positive");
  require(shape.embedding_dim > 0, "embedding_dim must be positive");
  require(shape.vocab_size > 0, "embedding vocab_size must be positive");
  require(stream != nullptr, "CUDA stream is null");
  require_dtype(dtype);
  (void)checked_mul(shape.vocab_size, shape.embedding_dim,
                    "embedding table shape overflow");
  const std::size_t total =
      checked_mul(shape.tokens, shape.embedding_dim, "embedding shape overflow");
  const int grid = checked_grid_for(total, "embedding grid dimension overflow");
  launch_by_dtype(dtype, [&]<typename T>() {
    embedding_kernel<T><<<grid, kBlockSize, 0, stream>>>(
        tokens, table, output, shape.tokens, shape.embedding_dim,
        shape.vocab_size);
  });
  check_launch("qwen35_embedding");
}

void qwen35_validate_token_ids(std::span<const std::int32_t> tokens,
                               std::size_t vocab_size) {
  require(!tokens.empty(), "token id span must not be empty");
  require(vocab_size > 0, "token vocab_size must be positive");
  for (const std::int32_t token : tokens) {
    require(token >= 0, "token id is negative");
    require(static_cast<std::size_t>(token) < vocab_size,
            "token id is out of range");
  }
}

void qwen35_rms_norm(const void* input, const void* weight, void* output,
                     RmsNormShape shape, float epsilon,
                     BrtDataType dtype, cudaStream_t stream) {
  require(input != nullptr, "rms_norm input pointer is null");
  require(weight != nullptr, "rms_norm weight pointer is null");
  require(output != nullptr, "rms_norm output pointer is null");
  require(shape.rows > 0, "rms_norm rows must be positive");
  require(shape.cols > 0, "rms_norm cols must be positive");
  require(std::isfinite(epsilon) && epsilon >= 0.0F,
          "rms_norm epsilon must be finite and non-negative");
  require(stream != nullptr, "CUDA stream is null");
  require_dtype(dtype);
  (void)checked_mul(shape.rows, shape.cols, "rms_norm shape overflow");
  const int grid =
      checked_block_count(shape.rows, "rms_norm grid dimension overflow");
  launch_by_dtype(dtype, [&]<typename T>() {
    rms_norm_kernel<T><<<grid, kBlockSize, kBlockSize * sizeof(float),
                         stream>>>(
        input, weight, output, shape.cols, epsilon);
  });
  check_launch("qwen35_rms_norm");
}

void qwen35_residual_add(const void* lhs, const void* rhs, void* output,
                         std::size_t elements, BrtDataType dtype,
                         cudaStream_t stream) {
  require(lhs != nullptr, "residual_add lhs pointer is null");
  require(rhs != nullptr, "residual_add rhs pointer is null");
  require(output != nullptr, "residual_add output pointer is null");
  require(elements > 0, "residual_add elements must be positive");
  require(stream != nullptr, "CUDA stream is null");
  require_dtype(dtype);
  const int grid =
      checked_grid_for(elements, "residual_add grid dimension overflow");
  launch_by_dtype(dtype, [&]<typename T>() {
    add_kernel<T><<<grid, kBlockSize, 0, stream>>>(lhs, rhs, output, elements);
  });
  check_launch("qwen35_residual_add");
}

void qwen35_qk_norm_rope(const void* input, const void* weight, void* output,
                         QkNormRopeShape shape, float epsilon,
                         BrtDataType dtype, cudaStream_t stream) {
  require(input != nullptr, "qk_norm_rope input pointer is null");
  require(weight != nullptr, "qk_norm_rope weight pointer is null");
  require(output != nullptr, "qk_norm_rope output pointer is null");
  require(shape.tokens > 0, "qk_norm_rope tokens must be positive");
  require(shape.heads > 0, "qk_norm_rope heads must be positive");
  require(shape.head_dim > 0, "qk_norm_rope head_dim must be positive");
  require(shape.rotary_dim > 0, "qk_norm_rope rotary_dim must be positive");
  require(shape.rotary_dim <= shape.head_dim,
          "qk_norm_rope rotary_dim exceeds head_dim");
  require(shape.rotary_dim % 2 == 0,
          "qk_norm_rope rotary_dim must be even");
  require(std::isfinite(shape.rope_base) && shape.rope_base > 0.0F,
          "qk_norm_rope rope_base must be finite and positive");
  require(std::isfinite(epsilon) && epsilon >= 0.0F,
          "qk_norm_rope epsilon must be finite and non-negative");
  require(shape.position_offset <=
              std::numeric_limits<std::size_t>::max() - (shape.tokens - 1),
          "qk_norm_rope position range overflow");
  require(stream != nullptr, "CUDA stream is null");
  require_dtype(dtype);
  const std::size_t vectors =
      checked_mul(shape.tokens, shape.heads, "qk_norm_rope vector overflow");
  (void)checked_mul(vectors, shape.head_dim, "qk_norm_rope shape overflow");
  const int grid =
      checked_block_count(vectors, "qk_norm_rope grid dimension overflow");
  launch_by_dtype(dtype, [&]<typename T>() {
    qk_norm_rope_kernel<T><<<grid, kBlockSize, kBlockSize * sizeof(float),
                            stream>>>(
        input, weight, output, shape.heads, shape.head_dim, shape.rotary_dim,
        shape.position_offset, shape.rope_base, epsilon);
  });
  check_launch("qwen35_qk_norm_rope");
}

void qwen35_sigmoid_gate(const void* values, const void* gates, void* output,
                         std::size_t elements, BrtDataType dtype,
                         cudaStream_t stream) {
  require(values != nullptr, "sigmoid_gate values pointer is null");
  require(gates != nullptr, "sigmoid_gate gates pointer is null");
  require(output != nullptr, "sigmoid_gate output pointer is null");
  require(elements > 0, "sigmoid_gate elements must be positive");
  require(stream != nullptr, "CUDA stream is null");
  require_dtype(dtype);
  const int grid =
      checked_grid_for(elements, "sigmoid_gate grid dimension overflow");
  launch_by_dtype(dtype, [&]<typename T>() {
    sigmoid_gate_kernel<T><<<grid, kBlockSize, 0, stream>>>(
        values, gates, output, elements);
  });
  check_launch("qwen35_sigmoid_gate");
}

void qwen35_swiglu(const void* gate, const void* up, void* output,
                   std::size_t elements, BrtDataType dtype,
                   cudaStream_t stream) {
  require(gate != nullptr, "swiglu gate pointer is null");
  require(up != nullptr, "swiglu up pointer is null");
  require(output != nullptr, "swiglu output pointer is null");
  require(elements > 0, "swiglu elements must be positive");
  require(stream != nullptr, "CUDA stream is null");
  require_dtype(dtype);
  const int grid = checked_grid_for(elements, "swiglu grid dimension overflow");
  launch_by_dtype(dtype, [&]<typename T>() {
    swiglu_kernel<T><<<grid, kBlockSize, 0, stream>>>(gate, up, output,
                                                      elements);
  });
  check_launch("qwen35_swiglu");
}

void qwen35_argmax(const float* logits, std::int32_t* output_index,
                   std::size_t elements, cudaStream_t stream) {
  require(logits != nullptr, "argmax logits pointer is null");
  require(output_index != nullptr, "argmax output pointer is null");
  require(elements > 0, "argmax elements must be positive");
  require(elements <= static_cast<std::size_t>(
                          std::numeric_limits<std::int32_t>::max()),
          "argmax elements exceed int32 output range");
  require(stream != nullptr, "CUDA stream is null");
  const std::size_t shared_bytes =
      kBlockSize * sizeof(float) + kBlockSize * sizeof(std::size_t);
  argmax_kernel<<<1, kBlockSize, shared_bytes, stream>>>(
      logits, output_index, elements);
  check_launch("qwen35_argmax");
}

}  // namespace brt::kernels
