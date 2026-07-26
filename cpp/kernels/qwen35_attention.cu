#include "qwen35_attention.cuh"
#include "qwen35_online_attention.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cmath>
#include <limits>
#include <string>

namespace brt::kernels {
namespace {

constexpr int kBlockSize = 256;

void require(bool condition, const char *message) {
  if (!condition) {
    throw Qwen35PrimitiveError(message);
  }
}

void require_dtype(BrtDataType dtype) {
  require(dtype == BRT_DTYPE_F32 || dtype == BRT_DTYPE_F16 ||
              dtype == BRT_DTYPE_BF16,
          "unsupported Qwen3.5 attention dtype");
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

int checked_grid_for(std::size_t blocks, const char *message) {
  require(blocks <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          message);
  return static_cast<int>(blocks);
}

void check_launch(const char *name) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw Qwen35PrimitiveError(std::string{name} +
                               " launch failed: " + cudaGetErrorString(error));
  }
}

template <typename T>
__device__ float load_as_float(const void *data, std::size_t index) {
  return static_cast<float>(static_cast<const T *>(data)[index]);
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

template <typename KernelLauncher>
void launch_by_dtype(BrtDataType dtype, KernelLauncher launcher) {
  if (dtype == BRT_DTYPE_F32) {
    launcher.template operator()<float>();
    return;
  }
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

std::size_t current_context_tokens(Qwen35AttentionShape shape) {
  return checked_add(shape.past_tokens, shape.tokens,
                     "attention context length overflow");
}

std::size_t hidden_size(Qwen35AttentionShape shape) {
  return checked_mul(shape.query_heads, shape.head_dim,
                     "attention hidden size overflow");
}

std::size_t kv_vector_size(Qwen35AttentionShape shape) {
  return checked_mul(shape.kv_heads, shape.head_dim,
                     "attention kv vector size overflow");
}

void validate_shape(Qwen35AttentionShape shape) {
  require(shape.tokens > 0, "attention tokens must be positive");
  require(shape.query_heads > 0, "attention query_heads must be positive");
  require(shape.kv_heads > 0, "attention kv_heads must be positive");
  require(shape.head_dim > 0, "attention head_dim must be positive");
  require(shape.max_context_tokens > 0,
          "attention max_context_tokens must be positive");
  require(shape.query_heads % shape.kv_heads == 0,
          "attention query_heads must be divisible by kv_heads");
  require(current_context_tokens(shape) <= shape.max_context_tokens,
          "attention context exceeds cache capacity");
  (void)checked_mul(shape.tokens, hidden_size(shape),
                    "attention output shape overflow");
  (void)checked_mul(std::size_t{2},
                    checked_mul(shape.max_context_tokens, kv_vector_size(shape),
                                "attention cache shape overflow"),
                    "attention cache shape overflow");
}

template <typename T>
__global__ void append_cache_kernel(const void *key, const void *value,
                                    float *kv_cache, std::size_t tokens,
                                    std::size_t kv_heads, std::size_t head_dim,
                                    std::size_t max_context_tokens,
                                    std::size_t past_tokens) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t kv_size = kv_heads * head_dim;
  const std::size_t total = tokens * kv_size;
  if (element >= total)
    return;

  const std::size_t token = element / kv_size;
  const std::size_t within_token = element - token * kv_size;
  const std::size_t cache_index =
      (past_tokens + token) * kv_size + within_token;
  const std::size_t cache_plane = max_context_tokens * kv_size;
  kv_cache[cache_index] = load_as_float<T>(key, element);
  kv_cache[cache_plane + cache_index] = load_as_float<T>(value, element);
}

template <typename T>
__global__ void
logits_kernel(const void *query, const float *kv_cache, float *logits,
              std::size_t tokens, std::size_t query_heads, std::size_t kv_heads,
              std::size_t head_dim, std::size_t max_context_tokens,
              std::size_t past_tokens) {
  extern __shared__ float shared[];
  const std::size_t pair = blockIdx.x;
  const std::size_t token = pair / query_heads;
  const std::size_t query_head = pair - token * query_heads;
  const std::size_t kv_head = query_head / (query_heads / kv_heads);
  const std::size_t current_context = past_tokens + tokens;
  const std::size_t visible_keys = past_tokens + token + 1;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  constexpr int kWarpSize = 32;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warp_count = blockDim.x / kWarpSize;

  for (std::size_t key_token = 0; key_token < visible_keys; ++key_token) {
    float thread_sum = 0.0F;
    for (std::size_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const std::size_t q_index =
          (token * query_heads + query_head) * head_dim + dim;
      const std::size_t k_index =
          (key_token * kv_heads + kv_head) * head_dim + dim;
      thread_sum += load_as_float<T>(query, q_index) * kv_cache[k_index];
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      thread_sum += __shfl_down_sync(0xFFFFFFFFU, thread_sum, offset);
    }
    if (lane == 0) {
      shared[warp] = thread_sum;
    }
    __syncthreads();
    if (warp == 0) {
      float block_sum = lane < warp_count ? shared[lane] : 0.0F;
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        block_sum += __shfl_down_sync(0xFFFFFFFFU, block_sum, offset);
      }
      if (lane == 0) {
        logits[(token * query_heads + query_head) * current_context +
               key_token] = block_sum * scale;
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void
output_kernel(const float *kv_cache, const void *gate, void *output,
              float *logits, std::size_t tokens, std::size_t query_heads,
              std::size_t kv_heads, std::size_t head_dim,
              std::size_t max_context_tokens, std::size_t past_tokens) {
  extern __shared__ float shared[];
  const std::size_t pair = blockIdx.x;
  const std::size_t token = pair / query_heads;
  const std::size_t query_head = pair - token * query_heads;
  const std::size_t kv_head = query_head / (query_heads / kv_heads);
  const std::size_t current_context = past_tokens + tokens;
  const std::size_t visible_keys = past_tokens + token + 1;
  const std::size_t logits_base =
      (token * query_heads + query_head) * current_context;
  const std::size_t kv_size = kv_heads * head_dim;
  const std::size_t value_plane = max_context_tokens * kv_size;

  if (threadIdx.x == 0) {
    float max_logit = -CUDART_INF_F;
    for (std::size_t key_token = 0; key_token < visible_keys; ++key_token) {
      max_logit = fmaxf(max_logit, logits[logits_base + key_token]);
    }
    double denom = 0.0;
    for (std::size_t key_token = 0; key_token < visible_keys; ++key_token) {
      denom += exp(static_cast<double>(logits[logits_base + key_token]) -
                   static_cast<double>(max_logit));
    }
    shared[0] = max_logit;
    shared[1] = static_cast<float>(denom);
  }
  __syncthreads();
  const float max_logit = shared[0];
  const float denom = shared[1];

  for (std::size_t key_token = threadIdx.x; key_token < visible_keys;
       key_token += blockDim.x) {
    logits[logits_base + key_token] =
        expf(logits[logits_base + key_token] - max_logit) / denom;
  }
  __syncthreads();

  for (std::size_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    double sum = 0.0;
    for (std::size_t key_token = 0; key_token < visible_keys; ++key_token) {
      const std::size_t value_index =
          value_plane + (key_token * kv_heads + kv_head) * head_dim + dim;
      sum += static_cast<double>(logits[logits_base + key_token]) *
             static_cast<double>(kv_cache[value_index]);
    }
    const std::size_t output_index =
        (token * query_heads + query_head) * head_dim + dim;
    const float gate_value = load_as_float<T>(gate, output_index);
    const float gated = static_cast<float>(sum) / (1.0F + expf(-gate_value));
    store_from_float<T>(output, output_index, gated);
  }
}

template <typename T>
__global__ void decode_logits_kernel(const void *query, const float *kv_cache,
                                     float *logits, std::size_t query_heads,
                                     std::size_t kv_heads, std::size_t head_dim,
                                     std::size_t current_context) {
  extern __shared__ float shared[];
  constexpr int kWarpSize = 32;
  const std::size_t flat_block = blockIdx.x;
  const std::size_t query_head = flat_block / current_context;
  const std::size_t key_token = flat_block - query_head * current_context;
  if (query_head >= query_heads)
    return;
  const std::size_t kv_head = query_head / (query_heads / kv_heads);

  float sum = 0.0F;
  for (std::size_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    sum += load_as_float<T>(query, query_head * head_dim + dim) *
           kv_cache[(key_token * kv_heads + kv_head) * head_dim + dim];
  }
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warp_count = blockDim.x / kWarpSize;
  if (lane == 0) {
    shared[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < warp_count ? shared[lane] : 0.0F;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      block_sum += __shfl_down_sync(0xFFFFFFFFU, block_sum, offset);
    }
    if (lane == 0) {
      logits[query_head * current_context + key_token] =
          block_sum * rsqrtf(static_cast<float>(head_dim));
    }
  }
}

__global__ void decode_softmax_kernel(float *logits, std::size_t query_heads,
                                      std::size_t current_context) {
  extern __shared__ float shared[];
  const std::size_t query_head = blockIdx.x;
  if (query_head >= query_heads)
    return;
  const std::size_t logits_base = query_head * current_context;

  if (threadIdx.x == 0) {
    float max_logit = -CUDART_INF_F;
    for (std::size_t key_token = 0; key_token < current_context; ++key_token) {
      max_logit = fmaxf(max_logit, logits[logits_base + key_token]);
    }
    double denom = 0.0;
    for (std::size_t key_token = 0; key_token < current_context; ++key_token) {
      denom += exp(static_cast<double>(logits[logits_base + key_token]) -
                   static_cast<double>(max_logit));
    }
    shared[0] = max_logit;
    shared[1] = static_cast<float>(denom);
  }
  __syncthreads();
  for (std::size_t key_token = threadIdx.x; key_token < current_context;
       key_token += blockDim.x) {
    logits[logits_base + key_token] =
        expf(logits[logits_base + key_token] - shared[0]) / shared[1];
  }
}

template <typename T>
__global__ void
decode_output_kernel(const float *kv_cache, const void *gate, void *output,
                     const float *probabilities, std::size_t query_heads,
                     std::size_t kv_heads, std::size_t head_dim,
                     std::size_t max_context_tokens,
                     std::size_t current_context, std::size_t dimension_tiles) {
  const std::size_t flat_block = blockIdx.x;
  const std::size_t query_head = flat_block / dimension_tiles;
  const std::size_t tile = flat_block - query_head * dimension_tiles;
  const std::size_t dim =
      tile * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (query_head >= query_heads || dim >= head_dim)
    return;
  const std::size_t kv_head = query_head / (query_heads / kv_heads);
  const std::size_t kv_size = kv_heads * head_dim;
  const std::size_t value_plane = max_context_tokens * kv_size;
  const std::size_t logits_base = query_head * current_context;

  double sum = 0.0;
  for (std::size_t key_token = 0; key_token < current_context; ++key_token) {
    const std::size_t value_index =
        value_plane + (key_token * kv_heads + kv_head) * head_dim + dim;
    sum += static_cast<double>(probabilities[logits_base + key_token]) *
           static_cast<double>(kv_cache[value_index]);
  }
  const std::size_t output_index = query_head * head_dim + dim;
  const float gate_value = load_as_float<T>(gate, output_index);
  const float gated = static_cast<float>(sum) / (1.0F + expf(-gate_value));
  store_from_float<T>(output, output_index, gated);
}

} // namespace

std::size_t qwen35_attention_workspace_floats(Qwen35AttentionShape shape) {
  validate_shape(shape);
  const std::size_t pairs = checked_mul(shape.tokens, shape.query_heads,
                                        "attention workspace shape overflow");
  return checked_mul(pairs, current_context_tokens(shape),
                     "attention workspace shape overflow");
}

std::size_t qwen35_attention_workspace_bytes(Qwen35AttentionShape shape) {
  const std::size_t floats = qwen35_attention_workspace_floats(shape);
  return checked_mul(floats, sizeof(float),
                     "attention workspace byte size overflow");
}

namespace {

void validate_launch_policy(Qwen35AttentionLaunchPolicy policy) {
  switch (policy.implementation) {
  case brt::Qwen35AttentionImplementation::materialized_reference:
  case brt::Qwen35AttentionImplementation::online_tiled:
    break;
  default:
    require(false, "unsupported Qwen3.5 attention implementation");
  }
  switch (policy.kv_cache_dtype) {
  case brt::Qwen35KvCacheDType::f32:
  case brt::Qwen35KvCacheDType::bf16:
    break;
  default:
    require(false, "unsupported Qwen3.5 KV cache dtype");
  }
  switch (policy.kv_cache_layout) {
  case brt::Qwen35KvCacheLayout::token_major:
  case brt::Qwen35KvCacheLayout::head_major:
    break;
  default:
    require(false, "unsupported Qwen3.5 KV cache layout");
  }
}

std::size_t kv_cache_element_bytes(brt::Qwen35KvCacheDType dtype) {
  switch (dtype) {
  case brt::Qwen35KvCacheDType::f32:
    return sizeof(float);
  case brt::Qwen35KvCacheDType::bf16:
    return sizeof(__nv_bfloat16);
  }
  require(false, "unsupported Qwen3.5 KV cache dtype");
  return 0;
}

void validate_attention_pointers(const void *query, const void *key,
                                 const void *value, const void *gate,
                                 void *output, void *kv_cache, void *workspace,
                                 std::size_t required_workspace_bytes,
                                 cudaStream_t stream) {
  require(query != nullptr, "attention query pointer is null");
  require(key != nullptr, "attention key pointer is null");
  require(value != nullptr, "attention value pointer is null");
  require(gate != nullptr, "attention gate pointer is null");
  require(output != nullptr, "attention output pointer is null");
  require(kv_cache != nullptr, "attention kv_cache pointer is null");
  require(required_workspace_bytes == 0 || workspace != nullptr,
          "attention workspace pointer is null");
  require(stream != nullptr, "CUDA stream is null");
}

} // namespace

std::size_t qwen35_attention_cache_bytes(Qwen35AttentionShape shape,
                                         Qwen35AttentionLaunchPolicy policy) {
  validate_shape(shape);
  validate_launch_policy(policy);
  const std::size_t elements =
      checked_mul(std::size_t{2},
                  checked_mul(shape.max_context_tokens, kv_vector_size(shape),
                              "attention cache shape overflow"),
                  "attention cache shape overflow");
  return checked_mul(elements, kv_cache_element_bytes(policy.kv_cache_dtype),
                     "attention cache byte size overflow");
}

std::size_t
qwen35_attention_workspace_bytes(Qwen35AttentionShape shape,
                                 Qwen35AttentionLaunchPolicy policy) {
  validate_launch_policy(policy);
  switch (policy.implementation) {
  case brt::Qwen35AttentionImplementation::materialized_reference:
    return qwen35_attention_workspace_bytes(shape);
  case brt::Qwen35AttentionImplementation::online_tiled:
    validate_shape(shape);
    return qwen35_online_attention_workspace_bytes(shape);
  }
  require(false, "unsupported Qwen3.5 attention implementation");
  return 0;
}

void qwen35_causal_attention_materialized(
    const void *query, const void *key, const void *value, const void *gate,
    void *output, void *kv_cache, std::size_t kv_cache_bytes, void *workspace,
    std::size_t workspace_bytes, Qwen35AttentionShape shape,
    BrtDataType activation_dtype, brt::Qwen35KvCacheDType cache_dtype,
    cudaStream_t stream) {
  require(cache_dtype == brt::Qwen35KvCacheDType::f32,
          "materialized attention requires an F32 KV cache");
  const Qwen35AttentionLaunchPolicy materialized_policy{
      .implementation =
          brt::Qwen35AttentionImplementation::materialized_reference,
      .kv_cache_dtype = cache_dtype,
      .kv_cache_layout = brt::Qwen35KvCacheLayout::token_major,
  };
  const std::size_t required_workspace =
      qwen35_attention_workspace_bytes(shape, materialized_policy);
  validate_attention_pointers(query, key, value, gate, output, kv_cache,
                              workspace, required_workspace, stream);
  require(kv_cache_bytes >=
              qwen35_attention_cache_bytes(shape, materialized_policy),
          "attention KV cache is too small");
  require(workspace_bytes >= required_workspace,
          "attention workspace is too small");
  require_dtype(activation_dtype);

  auto *const f32_kv_cache = static_cast<float *>(kv_cache);
  auto *const logits_workspace = static_cast<float *>(workspace);

  const std::size_t kv_elements =
      checked_mul(shape.tokens, kv_vector_size(shape),
                  "attention cache append shape overflow");
  const std::size_t attention_pairs = checked_mul(
      shape.tokens, shape.query_heads, "attention grid shape overflow");
  const int append_grid = checked_grid_for(
      kv_elements / kBlockSize + (kv_elements % kBlockSize == 0 ? 0 : 1),
      "attention cache append grid dimension overflow");
  const int pair_grid =
      checked_grid_for(attention_pairs, "attention grid dimension overflow");

  launch_by_dtype(activation_dtype, [&]<typename T>() {
    append_cache_kernel<T><<<append_grid, kBlockSize, 0, stream>>>(
        key, value, f32_kv_cache, shape.tokens, shape.kv_heads, shape.head_dim,
        shape.max_context_tokens, shape.past_tokens);
  });
  check_launch("qwen35_attention_append_cache");

  if (shape.tokens == 1) {
    const std::size_t current_context = current_context_tokens(shape);
    const int decode_logits_grid =
        checked_grid_for(checked_mul(shape.query_heads, current_context,
                                     "decode attention logits grid overflow"),
                         "decode attention logits grid overflow");
    launch_by_dtype(activation_dtype, [&]<typename T>() {
      decode_logits_kernel<T>
          <<<decode_logits_grid, kBlockSize, kBlockSize * sizeof(float),
             stream>>>(query, f32_kv_cache, logits_workspace, shape.query_heads,
                       shape.kv_heads, shape.head_dim, current_context);
    });
    check_launch("qwen35_decode_attention_logits");

    decode_softmax_kernel<<<static_cast<int>(shape.query_heads), kBlockSize,
                            2 * sizeof(float), stream>>>(
        logits_workspace, shape.query_heads, current_context);
    check_launch("qwen35_decode_attention_softmax");

    constexpr std::size_t kOutputTile = 64;
    const std::size_t dimension_tiles =
        shape.head_dim / kOutputTile +
        (shape.head_dim % kOutputTile == 0 ? 0 : 1);
    const int decode_output_grid =
        checked_grid_for(checked_mul(shape.query_heads, dimension_tiles,
                                     "decode attention output grid overflow"),
                         "decode attention output grid overflow");
    launch_by_dtype(activation_dtype, [&]<typename T>() {
      decode_output_kernel<T>
          <<<decode_output_grid, static_cast<int>(kOutputTile), 0, stream>>>(
              f32_kv_cache, gate, output, logits_workspace, shape.query_heads,
              shape.kv_heads, shape.head_dim, shape.max_context_tokens,
              current_context, dimension_tiles);
    });
    check_launch("qwen35_decode_attention_output");
    return;
  }

  launch_by_dtype(activation_dtype, [&]<typename T>() {
    logits_kernel<T>
        <<<pair_grid, kBlockSize, kBlockSize * sizeof(float), stream>>>(
            query, f32_kv_cache, logits_workspace, shape.tokens,
            shape.query_heads, shape.kv_heads, shape.head_dim,
            shape.max_context_tokens, shape.past_tokens);
  });
  check_launch("qwen35_attention_logits");

  launch_by_dtype(activation_dtype, [&]<typename T>() {
    output_kernel<T><<<pair_grid, kBlockSize, 2 * sizeof(float), stream>>>(
        f32_kv_cache, gate, output, logits_workspace, shape.tokens,
        shape.query_heads, shape.kv_heads, shape.head_dim,
        shape.max_context_tokens, shape.past_tokens);
  });
  check_launch("qwen35_attention_output");
}

void qwen35_causal_attention(const void *query, const void *key,
                             const void *value, const void *gate, void *output,
                             void *kv_cache, std::size_t kv_cache_bytes,
                             void *workspace, std::size_t workspace_bytes,
                             Qwen35AttentionShape shape,
                             BrtDataType activation_dtype,
                             Qwen35AttentionLaunchPolicy policy,
                             cudaStream_t stream) {
  const std::size_t required_workspace =
      qwen35_attention_workspace_bytes(shape, policy);
  validate_attention_pointers(query, key, value, gate, output, kv_cache,
                              workspace, required_workspace, stream);
  require(kv_cache_bytes >= qwen35_attention_cache_bytes(shape, policy),
          "attention KV cache is too small");
  require(workspace_bytes >= required_workspace,
          "attention workspace is too small");
  require_dtype(activation_dtype);

  switch (policy.implementation) {
  case brt::Qwen35AttentionImplementation::materialized_reference:
    require(policy.kv_cache_layout == brt::Qwen35KvCacheLayout::token_major,
            "materialized attention requires a token-major KV cache");
    qwen35_causal_attention_materialized(
        query, key, value, gate, output, kv_cache, kv_cache_bytes, workspace,
        workspace_bytes, shape, activation_dtype, policy.kv_cache_dtype,
        stream);
    return;
  case brt::Qwen35AttentionImplementation::online_tiled:
    require(workspace == nullptr && workspace_bytes == 0,
            "online tiled attention forbids logits workspace");
    require(policy.kv_cache_layout == brt::Qwen35KvCacheLayout::token_major,
            "online tiled attention requires a token-major KV cache");
    if (shape.tokens == 1) {
      qwen35_online_attention_decode(query, key, value, gate, output, kv_cache,
                                     kv_cache_bytes, shape, activation_dtype,
                                     policy.kv_cache_dtype, nullptr, stream);
      return;
    }
    require(qwen35_online_attention_prefill_supported(shape, activation_dtype,
                                                      policy),
            "unsupported online tiled prefill signature; select the "
            "materialized implementation before allocation");
    qwen35_online_attention_prefill(query, key, value, gate, output, kv_cache,
                                    kv_cache_bytes, shape, activation_dtype,
                                    policy.kv_cache_dtype, stream);
    return;
  }
  require(false, "unsupported Qwen3.5 attention implementation");
}

void qwen35_causal_attention(const void *query, const void *key,
                             const void *value, const void *gate, void *output,
                             float *kv_cache, float *logits_workspace,
                             std::size_t logits_workspace_floats,
                             Qwen35AttentionShape shape, BrtDataType dtype,
                             cudaStream_t stream) {
  const Qwen35AttentionLaunchPolicy policy{
      .implementation =
          brt::Qwen35AttentionImplementation::materialized_reference,
      .kv_cache_dtype = brt::Qwen35KvCacheDType::f32,
      .kv_cache_layout = brt::Qwen35KvCacheLayout::token_major,
  };
  qwen35_causal_attention(query, key, value, gate, output, kv_cache,
                          qwen35_attention_cache_bytes(shape, policy),
                          logits_workspace,
                          checked_mul(logits_workspace_floats, sizeof(float),
                                      "attention workspace byte size overflow"),
                          shape, dtype, policy, stream);
}

} // namespace brt::kernels
