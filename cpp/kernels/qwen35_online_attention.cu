#include "qwen35_online_attention.cuh"

#include "qwen35_primitives.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <limits>
#include <string>

namespace brt::kernels {
namespace {

constexpr int kWarpSize = 32;
constexpr int kQTile = 4;
constexpr int kKTile = 16;
constexpr int kBlockSize = kQTile * kWarpSize;
constexpr int kAppendBlockSize = 256;
constexpr int kModelHeadDim = 256;
constexpr int kModelQueryHeads = 16;
constexpr int kModelKvHeads = 4;
constexpr int kValuesPerLane = kModelHeadDim / kWarpSize;
constexpr std::size_t kMaxGridY = 65535;

bool multiplication_fits(std::size_t lhs, std::size_t rhs) noexcept {
  return lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs;
}

bool online_prefill_signature_supported(
    Qwen35AttentionShape shape, BrtDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype) noexcept {
  if (shape.tokens <= 1 || shape.query_heads == 0 || shape.kv_heads == 0 ||
      shape.head_dim == 0 || shape.head_dim > kModelHeadDim ||
      shape.max_context_tokens == 0 ||
      shape.query_heads % shape.kv_heads != 0 ||
      shape.past_tokens > shape.max_context_tokens ||
      shape.tokens > shape.max_context_tokens - shape.past_tokens) {
    return false;
  }
  if (activation_dtype != BRT_DTYPE_F32 && activation_dtype != BRT_DTYPE_BF16) {
    return false;
  }

  std::size_t cache_element_bytes = 0;
  switch (cache_dtype) {
  case Qwen35KvCacheDType::f32:
    cache_element_bytes = sizeof(float);
    break;
  case Qwen35KvCacheDType::bf16:
    cache_element_bytes = sizeof(__nv_bfloat16);
    break;
  default:
    return false;
  }

  const auto int_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  const std::size_t query_tiles =
      shape.tokens / kQTile + (shape.tokens % kQTile == 0 ? 0 : 1);
  if (shape.query_heads > int_limit || query_tiles > kMaxGridY ||
      !multiplication_fits(shape.kv_heads, shape.head_dim)) {
    return false;
  }
  const std::size_t kv_size = shape.kv_heads * shape.head_dim;
  if (!multiplication_fits(shape.tokens, kv_size) ||
      !multiplication_fits(shape.max_context_tokens, kv_size) ||
      !multiplication_fits(shape.query_heads, shape.head_dim)) {
    return false;
  }
  const std::size_t kv_elements = shape.tokens * kv_size;
  const std::size_t append_blocks =
      kv_elements / kAppendBlockSize +
      (kv_elements % kAppendBlockSize == 0 ? 0 : 1);
  if (append_blocks > int_limit ||
      !multiplication_fits(shape.tokens, shape.query_heads * shape.head_dim)) {
    return false;
  }
  const std::size_t cache_plane = shape.max_context_tokens * kv_size;
  return multiplication_fits(2, cache_plane) &&
         multiplication_fits(2 * cache_plane, cache_element_bytes);
}

bool online_decode_signature_supported(
    Qwen35AttentionShape shape, BrtDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype) noexcept {
  if (shape.tokens != 1 || shape.query_heads != kModelQueryHeads ||
      shape.kv_heads != kModelKvHeads || shape.head_dim != kModelHeadDim ||
      shape.max_context_tokens == 0 ||
      shape.past_tokens >= shape.max_context_tokens) {
    return false;
  }
  if (activation_dtype != BRT_DTYPE_F32 && activation_dtype != BRT_DTYPE_BF16) {
    return false;
  }

  std::size_t cache_element_bytes = 0;
  switch (cache_dtype) {
  case Qwen35KvCacheDType::f32:
    cache_element_bytes = sizeof(float);
    break;
  case Qwen35KvCacheDType::bf16:
    cache_element_bytes = sizeof(__nv_bfloat16);
    break;
  default:
    return false;
  }

  if (!multiplication_fits(shape.kv_heads, shape.head_dim) ||
      !multiplication_fits(shape.query_heads, shape.head_dim)) {
    return false;
  }
  const std::size_t kv_size = shape.kv_heads * shape.head_dim;
  if (!multiplication_fits(shape.max_context_tokens, kv_size)) {
    return false;
  }
  const std::size_t cache_plane = shape.max_context_tokens * kv_size;
  return multiplication_fits(2, cache_plane) &&
         multiplication_fits(2 * cache_plane, cache_element_bytes);
}

void require(bool condition, const char *message) {
  if (!condition) {
    throw Qwen35PrimitiveError(message);
  }
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, const char *message) {
  require(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs,
          message);
  return lhs * rhs;
}

int checked_grid_dimension(std::size_t value, const char *message) {
  require(value <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          message);
  return static_cast<int>(value);
}

void check_launch(const char *name) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw Qwen35PrimitiveError(std::string{name} +
                               " launch failed: " + cudaGetErrorString(error));
  }
}

template <typename T>
__device__ float load_as_float(const T *data, std::size_t index) {
  return static_cast<float>(data[index]);
}

template <>
__device__ float load_as_float(const __nv_bfloat16 *data, std::size_t index) {
  return __bfloat162float(data[index]);
}

template <typename T>
__device__ void store_from_float(T *data, std::size_t index, float value) {
  data[index] = static_cast<T>(value);
}

template <>
__device__ void store_from_float(__nv_bfloat16 *data, std::size_t index,
                                 float value) {
  data[index] = __float2bfloat16_rn(value);
}

template <typename ActivationT, typename CacheT>
__global__ void append_online_cache_kernel(
    const ActivationT *key, const ActivationT *value, CacheT *kv_cache,
    std::size_t tokens, std::size_t kv_heads, std::size_t head_dim,
    std::size_t max_context_tokens, std::size_t past_tokens) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t kv_size = kv_heads * head_dim;
  const std::size_t elements = tokens * kv_size;
  if (element >= elements) {
    return;
  }

  const std::size_t token = element / kv_size;
  const std::size_t within_token = element - token * kv_size;
  const std::size_t cache_index =
      (past_tokens + token) * kv_size + within_token;
  const std::size_t value_plane = max_context_tokens * kv_size;
  store_from_float(kv_cache, cache_index, load_as_float(key, element));
  store_from_float(kv_cache, value_plane + cache_index,
                   load_as_float(value, element));
}

template <typename ActivationT, typename CacheT>
__global__ void append_online_decode_cache_kernel(
    const ActivationT *key, const ActivationT *value, CacheT *kv_cache,
    std::size_t kv_size, std::size_t max_context_tokens,
    std::size_t host_position, const std::uint32_t *device_position) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= kv_size) {
    return;
  }

  const std::size_t position =
      device_position == nullptr ? host_position : *device_position;
  if (position >= max_context_tokens) {
    return;
  }

  const std::size_t cache_index = position * kv_size + element;
  const std::size_t value_plane = max_context_tokens * kv_size;
  store_from_float(kv_cache, cache_index, load_as_float(key, element));
  store_from_float(kv_cache, value_plane + cache_index,
                   load_as_float(value, element));
}

template <typename ActivationT, typename CacheT, int Warps>
__global__ __launch_bounds__(256, 1) void online_decode_model_kernel(
    const ActivationT *query, const ActivationT *gate, ActivationT *output,
    const CacheT *kv_cache, std::size_t max_context_tokens,
    std::size_t host_position, const std::uint32_t *device_position) {
  static_assert(Warps == 4 || Warps == 8);
  __shared__ float warp_max[Warps];
  __shared__ float warp_sum[Warps];
  __shared__ float warp_output[Warps][kModelHeadDim];

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const std::size_t position =
      device_position == nullptr ? host_position : *device_position;
  if (position >= max_context_tokens) {
    return;
  }

  const std::size_t context_tokens = position + 1;
  const int active_warps = context_tokens <= 128 ? 4 : Warps;
  const std::size_t query_head = blockIdx.x;
  const std::size_t kv_head = query_head / (kModelQueryHeads / kModelKvHeads);
  const std::size_t kv_size = kModelKvHeads * kModelHeadDim;
  const std::size_t value_plane = max_context_tokens * kv_size;
  const std::size_t query_base = query_head * kModelHeadDim;
  const float scale = rsqrtf(static_cast<float>(kModelHeadDim));

  float local_max = -CUDART_INF_F;
  float local_sum = 0.0F;
  float local_output[kValuesPerLane] = {};
  for (std::size_t key_token = static_cast<std::size_t>(warp);
       warp < active_warps && key_token < context_tokens;
       key_token += active_warps) {
    const std::size_t key_base =
        (key_token * kModelKvHeads + kv_head) * kModelHeadDim;
    float score = 0.0F;
#pragma unroll
    for (int component = 0; component < kValuesPerLane; ++component) {
      const int dim = lane + component * kWarpSize;
      score += load_as_float(query, query_base + dim) *
               load_as_float(kv_cache, key_base + dim);
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      score += __shfl_down_sync(0xFFFFFFFFU, score, offset);
    }
    score = __shfl_sync(0xFFFFFFFFU, score, 0) * scale;

    const float next_max = fmaxf(local_max, score);
    const float left_scale =
        local_max == -CUDART_INF_F ? 0.0F : expf(local_max - next_max);
    const float right_scale = expf(score - next_max);
    local_sum = local_sum * left_scale + right_scale;
    const std::size_t value_base = value_plane + key_base;
#pragma unroll
    for (int component = 0; component < kValuesPerLane; ++component) {
      const int dim = lane + component * kWarpSize;
      local_output[component] =
          local_output[component] * left_scale +
          load_as_float(kv_cache, value_base + dim) * right_scale;
    }
    local_max = next_max;
  }

  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_sum[warp] = local_sum;
  }
#pragma unroll
  for (int component = 0; component < kValuesPerLane; ++component) {
    const int dim = lane + component * kWarpSize;
    warp_output[warp][dim] = local_output[component];
  }
  __syncthreads();

  if (warp != 0) {
    return;
  }

  float merged_max = warp_max[0];
  float merged_sum = warp_sum[0];
  float merged_output[kValuesPerLane];
#pragma unroll
  for (int component = 0; component < kValuesPerLane; ++component) {
    const int dim = lane + component * kWarpSize;
    merged_output[component] = warp_output[0][dim];
  }

#pragma unroll
  for (int source_warp = 1; source_warp < Warps; ++source_warp) {
    if (source_warp >= active_warps) {
      break;
    }
    const float next_max = fmaxf(merged_max, warp_max[source_warp]);
    const float left_scale = expf(merged_max - next_max);
    const float right_scale = expf(warp_max[source_warp] - next_max);
    merged_sum = merged_sum * left_scale + warp_sum[source_warp] * right_scale;
#pragma unroll
    for (int component = 0; component < kValuesPerLane; ++component) {
      const int dim = lane + component * kWarpSize;
      merged_output[component] = merged_output[component] * left_scale +
                                 warp_output[source_warp][dim] * right_scale;
    }
    merged_max = next_max;
  }

#pragma unroll
  for (int component = 0; component < kValuesPerLane; ++component) {
    const int dim = lane + component * kWarpSize;
    const std::size_t output_index = query_base + dim;
    const float gate_value = load_as_float(gate, output_index);
    const float gated =
        (merged_output[component] / merged_sum) / (1.0F + expf(-gate_value));
    store_from_float(output, output_index, gated);
  }
}

template <typename ActivationT, typename CacheT, int HeadDim, int QueryHeads,
          int KvHeads>
__global__ __launch_bounds__(128, 2) void online_prefill_model_kernel(
    const ActivationT *query, const ActivationT *gate, ActivationT *output,
    const CacheT *kv_cache, std::size_t tokens, std::size_t max_context_tokens,
    std::size_t past_tokens) {
  static_assert(HeadDim == kModelHeadDim);
  static_assert(QueryHeads == kModelQueryHeads);
  static_assert(KvHeads == kModelKvHeads);
  const int lane = threadIdx.x % kWarpSize;
  const int query_in_tile = threadIdx.x / kWarpSize;
  const std::size_t query_token =
      static_cast<std::size_t>(blockIdx.y) * kQTile + query_in_tile;
  if (query_token >= tokens) {
    return;
  }

  const std::size_t query_head = blockIdx.x;
  const std::size_t kv_head =
      query_head / (static_cast<std::size_t>(QueryHeads) / KvHeads);
  const std::size_t visible_keys = past_tokens + query_token + 1;
  const std::size_t kv_size = static_cast<std::size_t>(KvHeads) * HeadDim;
  const std::size_t value_plane = max_context_tokens * kv_size;
  const std::size_t query_base =
      (query_token * QueryHeads + query_head) * HeadDim;
  const float scale = rsqrtf(static_cast<float>(HeadDim));

  float row_max = -CUDART_INF_F;
  float row_sum = 0.0F;
  float out_acc[kValuesPerLane] = {};

  for (std::size_t key_tile = 0; key_tile < visible_keys; key_tile += kKTile) {
    float scores[kKTile];
    float tile_max = -CUDART_INF_F;
    const std::size_t remaining_keys = visible_keys - key_tile;
    const std::size_t keys_in_tile =
        remaining_keys < static_cast<std::size_t>(kKTile)
            ? remaining_keys
            : static_cast<std::size_t>(kKTile);

#pragma unroll
    for (int key_in_tile = 0; key_in_tile < kKTile; ++key_in_tile) {
      float dot = 0.0F;
      if (static_cast<std::size_t>(key_in_tile) < keys_in_tile) {
        const std::size_t key_token = key_tile + key_in_tile;
        const std::size_t key_base = (key_token * KvHeads + kv_head) * HeadDim;
#pragma unroll
        for (int component = 0; component < kValuesPerLane; ++component) {
          const int dim = lane + component * kWarpSize;
          dot += load_as_float(query, query_base + dim) *
                 load_as_float(kv_cache, key_base + dim);
        }
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
          dot += __shfl_down_sync(0xFFFFFFFFU, dot, offset);
        }
        dot = __shfl_sync(0xFFFFFFFFU, dot, 0) * scale;
        tile_max = fmaxf(tile_max, dot);
      }
      scores[key_in_tile] = dot;
    }

    const float next_max = fmaxf(row_max, tile_max);
    const float old_scale =
        row_max == -CUDART_INF_F ? 0.0F : expf(row_max - next_max);
    row_sum *= old_scale;
#pragma unroll
    for (int component = 0; component < kValuesPerLane; ++component) {
      out_acc[component] *= old_scale;
    }

#pragma unroll
    for (int key_in_tile = 0; key_in_tile < kKTile; ++key_in_tile) {
      if (static_cast<std::size_t>(key_in_tile) >= keys_in_tile) {
        continue;
      }
      const std::size_t key_token = key_tile + key_in_tile;
      const std::size_t value_base =
          value_plane + (key_token * KvHeads + kv_head) * HeadDim;
      const float tile_scale = expf(scores[key_in_tile] - next_max);
      row_sum += tile_scale;
#pragma unroll
      for (int component = 0; component < kValuesPerLane; ++component) {
        const int dim = lane + component * kWarpSize;
        const float value_component = load_as_float(kv_cache, value_base + dim);
        out_acc[component] += tile_scale * value_component;
      }
    }
    row_max = next_max;
  }

#pragma unroll
  for (int component = 0; component < kValuesPerLane; ++component) {
    const int dim = lane + component * kWarpSize;
    const std::size_t output_index = query_base + dim;
    const float gate_value = load_as_float(gate, output_index);
    const float gated =
        (out_acc[component] / row_sum) / (1.0F + expf(-gate_value));
    store_from_float(output, output_index, gated);
  }
}

template <typename ActivationT, typename CacheT>
__global__ __launch_bounds__(128) void online_prefill_generic_kernel(
    const ActivationT *query, const ActivationT *gate, ActivationT *output,
    const CacheT *kv_cache, std::size_t tokens, std::size_t query_heads,
    std::size_t kv_heads, std::size_t head_dim, std::size_t max_context_tokens,
    std::size_t past_tokens) {
  const int lane = threadIdx.x % kWarpSize;
  const int query_in_tile = threadIdx.x / kWarpSize;
  const std::size_t query_token =
      static_cast<std::size_t>(blockIdx.y) * kQTile + query_in_tile;
  if (query_token >= tokens) {
    return;
  }

  const std::size_t query_head = blockIdx.x;
  const std::size_t kv_head = query_head / (query_heads / kv_heads);
  const std::size_t visible_keys = past_tokens + query_token + 1;
  const std::size_t kv_size = kv_heads * head_dim;
  const std::size_t value_plane = max_context_tokens * kv_size;
  const std::size_t query_base =
      (query_token * query_heads + query_head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float row_max = -CUDART_INF_F;
  float row_sum = 0.0F;
  float out_acc[kValuesPerLane] = {};

  for (std::size_t key_tile = 0; key_tile < visible_keys; key_tile += kKTile) {
    float scores[kKTile];
    float tile_max = -CUDART_INF_F;
    const std::size_t remaining_keys = visible_keys - key_tile;
    const std::size_t keys_in_tile =
        remaining_keys < static_cast<std::size_t>(kKTile)
            ? remaining_keys
            : static_cast<std::size_t>(kKTile);

#pragma unroll
    for (int key_in_tile = 0; key_in_tile < kKTile; ++key_in_tile) {
      float dot = 0.0F;
      if (static_cast<std::size_t>(key_in_tile) < keys_in_tile) {
        const std::size_t key_token = key_tile + key_in_tile;
        const std::size_t key_base =
            (key_token * kv_heads + kv_head) * head_dim;
#pragma unroll
        for (int component = 0; component < kValuesPerLane; ++component) {
          const std::size_t dim =
              static_cast<std::size_t>(lane + component * kWarpSize);
          if (dim < head_dim) {
            dot += load_as_float(query, query_base + dim) *
                   load_as_float(kv_cache, key_base + dim);
          }
        }
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
          dot += __shfl_down_sync(0xFFFFFFFFU, dot, offset);
        }
        dot = __shfl_sync(0xFFFFFFFFU, dot, 0) * scale;
        tile_max = fmaxf(tile_max, dot);
      }
      scores[key_in_tile] = dot;
    }

    const float next_max = fmaxf(row_max, tile_max);
    const float old_scale =
        row_max == -CUDART_INF_F ? 0.0F : expf(row_max - next_max);
    row_sum *= old_scale;
#pragma unroll
    for (int component = 0; component < kValuesPerLane; ++component) {
      out_acc[component] *= old_scale;
    }

#pragma unroll
    for (int key_in_tile = 0; key_in_tile < kKTile; ++key_in_tile) {
      if (static_cast<std::size_t>(key_in_tile) >= keys_in_tile) {
        continue;
      }
      const std::size_t key_token = key_tile + key_in_tile;
      const std::size_t value_base =
          value_plane + (key_token * kv_heads + kv_head) * head_dim;
      const float tile_scale = expf(scores[key_in_tile] - next_max);
      row_sum += tile_scale;
#pragma unroll
      for (int component = 0; component < kValuesPerLane; ++component) {
        const std::size_t dim =
            static_cast<std::size_t>(lane + component * kWarpSize);
        if (dim < head_dim) {
          const float value_component =
              load_as_float(kv_cache, value_base + dim);
          out_acc[component] += tile_scale * value_component;
        }
      }
    }
    row_max = next_max;
  }

#pragma unroll
  for (int component = 0; component < kValuesPerLane; ++component) {
    const std::size_t dim =
        static_cast<std::size_t>(lane + component * kWarpSize);
    if (dim < head_dim) {
      const std::size_t output_index = query_base + dim;
      const float gate_value = load_as_float(gate, output_index);
      const float gated =
          (out_acc[component] / row_sum) / (1.0F + expf(-gate_value));
      store_from_float(output, output_index, gated);
    }
  }
}

template <typename ActivationT, typename CacheT>
void launch_online_prefill(const void *query, const void *key,
                           const void *value, const void *gate, void *output,
                           void *kv_cache, Qwen35AttentionShape shape,
                           cudaStream_t stream) {
  const std::size_t kv_elements =
      checked_mul(shape.tokens,
                  checked_mul(shape.kv_heads, shape.head_dim,
                              "online attention shape overflow"),
                  "online attention shape overflow");
  const int append_grid = checked_grid_dimension(
      (kv_elements + kAppendBlockSize - 1) / kAppendBlockSize,
      "online attention append grid dimension overflow");
  append_online_cache_kernel<ActivationT, CacheT>
      <<<append_grid, kAppendBlockSize, 0, stream>>>(
          static_cast<const ActivationT *>(key),
          static_cast<const ActivationT *>(value),
          static_cast<CacheT *>(kv_cache), shape.tokens, shape.kv_heads,
          shape.head_dim, shape.max_context_tokens, shape.past_tokens);
  check_launch("qwen35_online_attention_append_cache");

  const dim3 grid{static_cast<unsigned int>(checked_grid_dimension(
                      shape.query_heads,
                      "online attention query-head grid dimension overflow")),
                  static_cast<unsigned int>(checked_grid_dimension(
                      (shape.tokens + kQTile - 1) / kQTile,
                      "online attention query-tile grid dimension overflow")),
                  1};
  if (shape.head_dim == kModelHeadDim &&
      shape.query_heads == kModelQueryHeads &&
      shape.kv_heads == kModelKvHeads) {
    online_prefill_model_kernel<ActivationT, CacheT, kModelHeadDim,
                                kModelQueryHeads, kModelKvHeads>
        <<<grid, kBlockSize, 0, stream>>>(
            static_cast<const ActivationT *>(query),
            static_cast<const ActivationT *>(gate),
            static_cast<ActivationT *>(output),
            static_cast<const CacheT *>(kv_cache), shape.tokens,
            shape.max_context_tokens, shape.past_tokens);
    check_launch("qwen35_online_attention_prefill_sm120_hd256");
    return;
  }

  online_prefill_generic_kernel<ActivationT, CacheT>
      <<<grid, kBlockSize, 0, stream>>>(
          static_cast<const ActivationT *>(query),
          static_cast<const ActivationT *>(gate),
          static_cast<ActivationT *>(output),
          static_cast<const CacheT *>(kv_cache), shape.tokens,
          shape.query_heads, shape.kv_heads, shape.head_dim,
          shape.max_context_tokens, shape.past_tokens);
  check_launch("qwen35_online_attention_prefill_generic");
}

template <typename ActivationT, typename CacheT>
void launch_online_decode(const void *query, const void *key, const void *value,
                          const void *gate, void *output, void *kv_cache,
                          Qwen35AttentionShape shape,
                          const std::uint32_t *device_position,
                          cudaStream_t stream) {
  constexpr std::size_t kv_size = kModelKvHeads * kModelHeadDim;
  constexpr int append_grid =
      static_cast<int>((kv_size + kAppendBlockSize - 1) / kAppendBlockSize);
  append_online_decode_cache_kernel<ActivationT, CacheT>
      <<<append_grid, kAppendBlockSize, 0, stream>>>(
          static_cast<const ActivationT *>(key),
          static_cast<const ActivationT *>(value),
          static_cast<CacheT *>(kv_cache), kv_size, shape.max_context_tokens,
          shape.past_tokens, device_position);
  check_launch("qwen35_online_attention_decode_append_cache");

  const int query_head_grid = checked_grid_dimension(
      shape.query_heads, "online decode query-head grid dimension overflow");
  if (device_position == nullptr && shape.past_tokens + 1 <= 128) {
    online_decode_model_kernel<ActivationT, CacheT, 4>
        <<<query_head_grid, 4 * kWarpSize, 0, stream>>>(
            static_cast<const ActivationT *>(query),
            static_cast<const ActivationT *>(gate),
            static_cast<ActivationT *>(output),
            static_cast<const CacheT *>(kv_cache), shape.max_context_tokens,
            shape.past_tokens, device_position);
  } else {
    online_decode_model_kernel<ActivationT, CacheT, 8>
        <<<query_head_grid, 8 * kWarpSize, 0, stream>>>(
            static_cast<const ActivationT *>(query),
            static_cast<const ActivationT *>(gate),
            static_cast<ActivationT *>(output),
            static_cast<const CacheT *>(kv_cache), shape.max_context_tokens,
            shape.past_tokens, device_position);
  }
  check_launch("qwen35_online_attention_decode_sm120_hd256");
}

template <typename ActivationT>
void launch_by_cache_dtype(const void *query, const void *key,
                           const void *value, const void *gate, void *output,
                           void *kv_cache, Qwen35AttentionShape shape,
                           Qwen35KvCacheDType cache_dtype,
                           cudaStream_t stream) {
  switch (cache_dtype) {
  case Qwen35KvCacheDType::f32:
    launch_online_prefill<ActivationT, float>(query, key, value, gate, output,
                                              kv_cache, shape, stream);
    return;
  case Qwen35KvCacheDType::bf16:
    launch_online_prefill<ActivationT, __nv_bfloat16>(
        query, key, value, gate, output, kv_cache, shape, stream);
    return;
  }
  require(false, "unsupported online attention KV cache dtype");
}

template <typename ActivationT>
void launch_decode_by_cache_dtype(const void *query, const void *key,
                                  const void *value, const void *gate,
                                  void *output, void *kv_cache,
                                  Qwen35AttentionShape shape,
                                  Qwen35KvCacheDType cache_dtype,
                                  const std::uint32_t *device_position,
                                  cudaStream_t stream) {
  switch (cache_dtype) {
  case Qwen35KvCacheDType::f32:
    launch_online_decode<ActivationT, float>(query, key, value, gate, output,
                                             kv_cache, shape, device_position,
                                             stream);
    return;
  case Qwen35KvCacheDType::bf16:
    launch_online_decode<ActivationT, __nv_bfloat16>(query, key, value, gate,
                                                     output, kv_cache, shape,
                                                     device_position, stream);
    return;
  }
  require(false, "unsupported online attention KV cache dtype");
}

} // namespace

bool qwen35_online_attention_prefill_supported(
    Qwen35AttentionShape shape, BrtDataType activation_dtype,
    Qwen35AttentionLaunchPolicy policy) noexcept {
  if (policy.implementation != Qwen35AttentionImplementation::online_tiled ||
      policy.kv_cache_layout != Qwen35KvCacheLayout::token_major) {
    return false;
  }
  return online_prefill_signature_supported(shape, activation_dtype,
                                            policy.kv_cache_dtype);
}

std::size_t
qwen35_online_attention_workspace_bytes(Qwen35AttentionShape shape) noexcept {
  (void)shape;
  return 0;
}

void qwen35_online_attention_prefill(
    const void *query, const void *key, const void *value, const void *gate,
    void *output, void *kv_cache, std::size_t kv_cache_bytes,
    Qwen35AttentionShape shape, BrtDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype, cudaStream_t stream) {
  require(query != nullptr, "online attention query pointer is null");
  require(key != nullptr, "online attention key pointer is null");
  require(value != nullptr, "online attention value pointer is null");
  require(gate != nullptr, "online attention gate pointer is null");
  require(output != nullptr, "online attention output pointer is null");
  require(kv_cache != nullptr, "online attention KV cache pointer is null");
  require(stream != nullptr, "online attention CUDA stream is null");
  require(
      online_prefill_signature_supported(shape, activation_dtype, cache_dtype),
      "unsupported online attention prefill signature; select the "
      "materialized implementation before allocation");

  std::size_t cache_element_bytes = 0;
  switch (cache_dtype) {
  case Qwen35KvCacheDType::f32:
    cache_element_bytes = sizeof(float);
    break;
  case Qwen35KvCacheDType::bf16:
    cache_element_bytes = sizeof(__nv_bfloat16);
    break;
  default:
    require(false, "unsupported online attention KV cache dtype");
  }
  const std::size_t required_cache_bytes = checked_mul(
      checked_mul(
          2,
          checked_mul(shape.max_context_tokens,
                      checked_mul(shape.kv_heads, shape.head_dim,
                                  "online attention cache shape overflow"),
                      "online attention cache shape overflow"),
          "online attention cache shape overflow"),
      cache_element_bytes, "online attention cache byte size overflow");
  require(kv_cache_bytes >= required_cache_bytes,
          "online attention KV cache is too small");

  if (activation_dtype == BRT_DTYPE_F32) {
    launch_by_cache_dtype<float>(query, key, value, gate, output, kv_cache,
                                 shape, cache_dtype, stream);
    return;
  }
  launch_by_cache_dtype<__nv_bfloat16>(query, key, value, gate, output,
                                       kv_cache, shape, cache_dtype, stream);
}

void qwen35_online_attention_decode(
    const void *query, const void *key, const void *value, const void *gate,
    void *output, void *kv_cache, std::size_t kv_cache_bytes,
    Qwen35AttentionShape shape, BrtDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype, const std::uint32_t *device_position,
    cudaStream_t stream) {
  require(query != nullptr, "online decode query pointer is null");
  require(key != nullptr, "online decode key pointer is null");
  require(value != nullptr, "online decode value pointer is null");
  require(gate != nullptr, "online decode gate pointer is null");
  require(output != nullptr, "online decode output pointer is null");
  require(kv_cache != nullptr, "online decode KV cache pointer is null");
  require(stream != nullptr, "online decode CUDA stream is null");
  require(
      online_decode_signature_supported(shape, activation_dtype, cache_dtype),
      "unsupported online attention decode signature; select the "
      "materialized implementation before allocation");

  const std::size_t cache_element_bytes = cache_dtype == Qwen35KvCacheDType::f32
                                              ? sizeof(float)
                                              : sizeof(__nv_bfloat16);
  const std::size_t required_cache_bytes = checked_mul(
      checked_mul(2,
                  checked_mul(shape.max_context_tokens,
                              checked_mul(shape.kv_heads, shape.head_dim,
                                          "online decode cache shape overflow"),
                              "online decode cache shape overflow"),
                  "online decode cache shape overflow"),
      cache_element_bytes, "online decode cache byte size overflow");
  require(kv_cache_bytes >= required_cache_bytes,
          "online decode KV cache is too small");

  if (activation_dtype == BRT_DTYPE_F32) {
    launch_decode_by_cache_dtype<float>(query, key, value, gate, output,
                                        kv_cache, shape, cache_dtype,
                                        device_position, stream);
    return;
  }
  launch_decode_by_cache_dtype<__nv_bfloat16>(query, key, value, gate, output,
                                              kv_cache, shape, cache_dtype,
                                              device_position, stream);
}

} // namespace brt::kernels
