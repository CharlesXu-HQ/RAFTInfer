#include "qwen35_executor.hpp"

#include "cublaslt_matmul.hpp"
#include "cuda_graph_decode.hpp"

#include "../kernels/qwen35_attention.cuh"
#include "../kernels/qwen35_delta.cuh"
#include "../kernels/qwen35_online_attention.cuh"
#include "../kernels/qwen35_primitives.cuh"

#include <brt/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace brt {
namespace {

constexpr std::size_t kMaxPrefillTokens = 512;
constexpr std::array<std::size_t, 6> kBuckets{1, 2, 4, 17, 128, 512};
constexpr std::size_t kMatmulWorkspaceBudget = 4U * 1024U * 1024U;

__global__ void commit_decode_result(std::int32_t *next_token,
                                     const std::int32_t *result,
                                     std::int32_t *results,
                                     std::uint32_t *position) {
  const std::uint32_t current_position = *position;
  const std::int32_t token = *result;
  results[current_position] = token;
  *next_token = token;
  *position = current_position + 1;
}

__global__ void fill_delta_tuning_input(void *input, std::size_t elements,
                                        BrtDataType dtype) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const int lane = static_cast<int>((index * 17U) % 37U) - 18;
  const float value = 0.0075F * static_cast<float>(lane);
  if (dtype == BRT_DTYPE_F32) {
    static_cast<float *>(input)[index] = value;
  } else if (dtype == BRT_DTYPE_F16) {
    static_cast<__half *>(input)[index] = __float2half_rn(value);
  } else if (dtype == BRT_DTYPE_BF16) {
    static_cast<__nv_bfloat16 *>(input)[index] = __float2bfloat16_rn(value);
  }
}

void require(bool condition, const char *message) {
  if (!condition)
    throw Qwen35ExecutorError(message);
}

struct CudaHostInt32Deleter {
  void operator()(std::int32_t *ptr) const noexcept {
    if (ptr != nullptr)
      (void)cudaFreeHost(ptr);
  }
};

using CudaHostInt32 = std::unique_ptr<std::int32_t, CudaHostInt32Deleter>;

std::size_t align_up(std::size_t value, std::size_t alignment) {
  require(alignment != 0 && (alignment & (alignment - 1)) == 0,
          "alignment must be a power of two");
  require(value <= std::numeric_limits<std::size_t>::max() - (alignment - 1),
          "workspace alignment overflow");
  return (value + alignment - 1) & ~(alignment - 1);
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

std::size_t dtype_size(BrtDataType dtype) {
  if (dtype == BRT_DTYPE_F32)
    return 4;
  if (dtype == BRT_DTYPE_F16 || dtype == BRT_DTYPE_BF16)
    return 2;
  throw Qwen35ExecutorError("Qwen3.5 executor weights must be F16 or BF16");
}

BrtDataType dtype_from_weight(model::CudaWeightType type) {
  if (type == model::CudaWeightType::f32)
    return BRT_DTYPE_F32;
  if (type == model::CudaWeightType::f16)
    return BRT_DTYPE_F16;
  if (type == model::CudaWeightType::bf16)
    return BRT_DTYPE_BF16;
  throw Qwen35ExecutorError("unsupported Qwen3.5 CUDA weight dtype");
}

void check_cuda(cudaError_t error, const char *message) {
  if (error != cudaSuccess) {
    throw Qwen35ExecutorError(std::string{message} + ": " +
                              cudaGetErrorString(error));
  }
}

class DeviceGuard {
public:
  explicit DeviceGuard(int target) {
    check_cuda(cudaGetDevice(&previous_),
               "Qwen3.5 executor device query failed");
    if (previous_ != target) {
      check_cuda(cudaSetDevice(target),
                 "Qwen3.5 executor device selection failed");
      changed_ = true;
    }
  }
  ~DeviceGuard() noexcept {
    if (changed_)
      (void)cudaSetDevice(previous_);
  }

  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard &operator=(const DeviceGuard &) = delete;

private:
  int previous_{};
  bool changed_{};
};

class CudaEvent {
public:
  explicit CudaEvent(const char *message) {
    check_cuda(cudaEventCreate(&event_), message);
  }

  ~CudaEvent() noexcept {
    if (event_ != nullptr)
      (void)cudaEventDestroy(event_);
  }

  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;

  cudaEvent_t get() const noexcept { return event_; }

private:
  cudaEvent_t event_{};
};

void *allocate(WorkspaceArena &arena, std::size_t bytes,
               std::size_t alignment) {
  return arena.allocate(bytes == 0 ? 1 : align_up(bytes, alignment), alignment);
}

struct LinearPlans {
  std::shared_ptr<CublasLtMatmulPlan> qkv;
  std::shared_ptr<CublasLtMatmulPlan> beta;
  std::shared_ptr<CublasLtMatmulPlan> alpha;
  std::shared_ptr<CublasLtMatmulPlan> gate;
  std::shared_ptr<CublasLtMatmulPlan> output;
};

struct FullPlans {
  std::shared_ptr<CublasLtMatmulPlan> query_gate;
  std::shared_ptr<CublasLtMatmulPlan> key;
  std::shared_ptr<CublasLtMatmulPlan> value;
  std::shared_ptr<CublasLtMatmulPlan> output;
};

struct LayerPlans {
  std::shared_ptr<CublasLtMatmulPlan> ffn_gate;
  std::shared_ptr<CublasLtMatmulPlan> ffn_up;
  std::shared_ptr<CublasLtMatmulPlan> ffn_down;
  std::unique_ptr<LinearPlans> linear;
  std::unique_ptr<FullPlans> full;
};

struct BucketPlans {
  std::size_t tokens{};
  std::vector<LayerPlans> layers;
  std::shared_ptr<CublasLtMatmulPlan> lm_head;
};

struct PlanCacheEntry {
  CublasLtMatmulConfig config;
  std::shared_ptr<CublasLtMatmulPlan> plan;
};

bool same_matmul_config(const CublasLtMatmulConfig &lhs,
                        const CublasLtMatmulConfig &rhs) {
  return lhs.shape.m == rhs.shape.m && lhs.shape.n == rhs.shape.n &&
         lhs.shape.k == rhs.shape.k &&
         lhs.shape.transpose_input == rhs.shape.transpose_input &&
         lhs.shape.transpose_weight == rhs.shape.transpose_weight &&
         lhs.input_dtype == rhs.input_dtype &&
         lhs.weight_dtype == rhs.weight_dtype &&
         lhs.output_dtype == rhs.output_dtype &&
         lhs.input_order == rhs.input_order &&
         lhs.weight_order == rhs.weight_order &&
         lhs.output_order == rhs.output_order &&
         lhs.workspace_budget_bytes == rhs.workspace_budget_bytes;
}

bool release_tuning_bucket(std::size_t bucket) {
  return bucket == 1 || bucket == 128 || bucket == 512;
}

float median_ms(std::vector<float> values) {
  require(!values.empty(), "median requires at least one value");
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

bool close_enough(float actual, float expected) {
  const float absolute = std::fabs(actual - expected);
  const float relative = absolute / std::max(std::fabs(expected), 1.0e-6F);
  return absolute <= 2.0e-2F || relative <= 2.0e-2F;
}

CublasLtMatmulConfig matmul_config(std::size_t m, std::size_t n, std::size_t k,
                                   BrtDataType activation_dtype,
                                   BrtDataType weight_dtype) {
  return CublasLtMatmulConfig{
      .shape = CublasLtMatmulShape{.m = m,
                                   .n = n,
                                   .k = k,
                                   .transpose_input = false,
                                   .transpose_weight = true},
      .input_dtype = weight_dtype,
      .weight_dtype = weight_dtype,
      .output_dtype = activation_dtype,
      .input_order = CublasLtMatrixOrder::RowMajor,
      .weight_order = CublasLtMatrixOrder::RowMajor,
      .output_order = CublasLtMatrixOrder::RowMajor,
      .workspace_budget_bytes = kMatmulWorkspaceBudget,
  };
}

std::size_t attention_cache_bytes(kernels::Qwen35AttentionShape shape,
                                  kernels::Qwen35AttentionLaunchPolicy policy) {
  try {
    return kernels::qwen35_attention_cache_bytes(shape, policy);
  } catch (const kernels::Qwen35PrimitiveError &error) {
    throw Qwen35ExecutorError(error.what());
  }
}

std::size_t
attention_workspace_bytes(kernels::Qwen35AttentionShape shape,
                          kernels::Qwen35AttentionLaunchPolicy policy) {
  try {
    return kernels::qwen35_attention_workspace_bytes(shape, policy);
  } catch (const kernels::Qwen35PrimitiveError &error) {
    throw Qwen35ExecutorError(error.what());
  }
}

std::size_t
max_attention_workspace(const model::Qwen35Config &config, std::size_t tokens,
                        std::size_t max_context,
                        kernels::Qwen35AttentionLaunchPolicy policy) {
  if (config.full_attention_head_count == 0)
    return 0;
  return attention_workspace_bytes(
      kernels::Qwen35AttentionShape{
          .tokens = tokens,
          .query_heads = config.full_attention_head_count,
          .kv_heads = config.full_attention_kv_head_count,
          .head_dim = config.full_attention_head_dimension,
          .max_context_tokens = max_context,
          .past_tokens = max_context - tokens},
      policy);
}

std::size_t linear_qkv_width(const model::Qwen35Config &config) {
  const std::size_t key_width =
      checked_mul(config.linear_key_head_count, config.linear_head_dimension,
                  "linear key width overflow");
  const std::size_t value_width =
      checked_mul(config.linear_value_head_count, config.linear_head_dimension,
                  "linear value width overflow");
  return checked_add(checked_mul(2, key_width, "linear qkv width overflow"),
                     value_width, "linear qkv width overflow");
}

void validate_config(const model::Qwen35Config &config,
                     std::size_t max_context) {
  require(config.vocabulary_size > 0, "vocabulary_size must be positive");
  require(config.hidden_size > 0, "hidden_size must be positive");
  require(config.intermediate_size > 0, "intermediate_size must be positive");
  require(config.context_length > 0, "context_length must be positive");
  require(max_context > 0, "max_context must be positive");
  require(max_context <= config.context_length,
          "max_context exceeds configured context length");
  require(!config.blocks.empty(), "Qwen3.5 config must contain blocks");
  require(config.rms_norm_epsilon >= 0.0F,
          "rms_norm_epsilon must be non-negative");
  require(config.rope_frequency_base > 0.0F,
          "rope frequency base must be positive");
  require(config.linear_convolution_width > 0,
          "linear convolution width must be positive");
  require(config.linear_key_head_count > 0,
          "linear key head count must be positive");
  require(config.linear_value_head_count > 0,
          "linear value head count must be positive");
  require(config.linear_head_dimension > 0,
          "linear head dimension must be positive");
  require(config.full_attention_head_count > 0,
          "full attention head count must be positive");
  require(config.full_attention_kv_head_count > 0,
          "full attention KV head count must be positive");
  require(config.full_attention_head_dimension > 0,
          "full attention head dimension must be positive");
  require(config.rotary_dimension > 0, "rotary dimension must be positive");
  require(config.rotary_dimension <= config.full_attention_head_dimension,
          "rotary dimension exceeds full attention head dimension");
  require(config.rotary_dimension % 2 == 0, "rotary dimension must be even");
  require(config.full_attention_head_count %
                  config.full_attention_kv_head_count ==
              0,
          "full attention heads must be divisible by KV heads");
  const std::size_t full_width = checked_mul(
      config.full_attention_head_count, config.full_attention_head_dimension,
      "full attention width overflow");
  require(full_width == config.hidden_size,
          "full attention query width must equal hidden_size");
  const std::size_t linear_value_width =
      checked_mul(config.linear_value_head_count, config.linear_head_dimension,
                  "linear value width overflow");
  require(linear_value_width == config.hidden_size,
          "linear value width must equal hidden_size");
  require(config.linear_value_head_count % config.linear_key_head_count == 0,
          "linear value heads must be divisible by key heads");
  for (std::size_t i = 0; i < config.blocks.size(); ++i) {
    require(config.blocks[i].index == i,
            "Qwen3.5 block indices must be contiguous");
    require(config.blocks[i].kind == model::Qwen35BlockKind::full_attention ||
                config.blocks[i].kind ==
                    model::Qwen35BlockKind::linear_attention,
            "Qwen3.5 block kind is invalid");
  }
}

void validate_execution_policy(Qwen35ExecutionPolicy policy) {
  switch (policy.attention) {
  case Qwen35AttentionImplementation::materialized_reference:
  case Qwen35AttentionImplementation::online_tiled:
    break;
  default:
    require(false, "unsupported Qwen3.5 attention implementation");
  }
  switch (policy.kv_cache) {
  case Qwen35KvCacheDType::f32:
  case Qwen35KvCacheDType::bf16:
    break;
  default:
    require(false, "unsupported Qwen3.5 KV cache dtype");
  }
  switch (policy.kv_cache_layout) {
  case Qwen35KvCacheLayout::token_major:
  case Qwen35KvCacheLayout::head_major:
    break;
  default:
    require(false, "unsupported Qwen3.5 KV cache layout");
  }
}

kernels::Qwen35AttentionLaunchPolicy
attention_launch_policy(Qwen35ExecutionPolicy policy) noexcept {
  return kernels::Qwen35AttentionLaunchPolicy{
      .implementation = policy.attention,
      .kv_cache_dtype = policy.kv_cache,
      .kv_cache_layout = policy.kv_cache_layout,
  };
}

bool online_decode_supported(const model::Qwen35Config &config,
                             Qwen35ExecutionPolicy policy) noexcept {
  if (config.full_attention_head_count != 16 ||
      config.full_attention_kv_head_count != 4 ||
      config.full_attention_head_dimension != 256) {
    return false;
  }
  switch (policy.kv_cache_layout) {
  case Qwen35KvCacheLayout::token_major:
  case Qwen35KvCacheLayout::head_major:
    return true;
  default:
    return false;
  }
}

Qwen35ExecutionPolicy
resolve_execution_policy(const model::Qwen35Config &config,
                         std::size_t max_context,
                         Qwen35ExecutionPolicy requested) {
  validate_config(config, max_context);
  validate_execution_policy(requested);
  if (requested.attention ==
      Qwen35AttentionImplementation::materialized_reference) {
    require(requested.kv_cache == Qwen35KvCacheDType::f32,
            "materialized attention requires an F32 KV cache");
    require(requested.kv_cache_layout == Qwen35KvCacheLayout::token_major,
            "materialized attention requires a token-major KV cache");
    return requested;
  }

  bool supported = online_decode_supported(config, requested);
  if (supported && max_context > 1) {
    const std::size_t tokens =
        std::min<std::size_t>(kMaxPrefillTokens, max_context);
    supported = kernels::qwen35_online_attention_prefill_supported(
        kernels::Qwen35AttentionShape{
            .tokens = tokens,
            .query_heads = config.full_attention_head_count,
            .kv_heads = config.full_attention_kv_head_count,
            .head_dim = config.full_attention_head_dimension,
            .max_context_tokens = max_context,
            .past_tokens = max_context - tokens,
        },
        BRT_DTYPE_F32, attention_launch_policy(requested));
  }
  if (supported)
    return requested;

  requested.attention = Qwen35AttentionImplementation::materialized_reference;
  requested.kv_cache = Qwen35KvCacheDType::f32;
  requested.kv_cache_layout = Qwen35KvCacheLayout::token_major;
  return requested;
}

std::size_t workspace_estimate(const model::Qwen35Config &config,
                               std::size_t max_context,
                               Qwen35ExecutionPolicy requested_policy) {
  const auto policy =
      resolve_execution_policy(config, max_context, requested_policy);
  const auto attention_policy = attention_launch_policy(policy);
  const std::size_t element = sizeof(float);
  const std::size_t max_tokens =
      std::min<std::size_t>(kMaxPrefillTokens, max_context);
  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t qkv_width = linear_qkv_width(config);
  const std::size_t linear_beta_width = config.linear_value_head_count;
  const std::size_t linear_alpha_width = config.linear_value_head_count;
  const std::size_t linear_gate_width =
      checked_mul(config.linear_value_head_count, config.linear_head_dimension,
                  "linear gate width overflow");
  const std::size_t linear_pack_width = checked_add(
      checked_add(qkv_width, linear_beta_width, "linear pack width overflow"),
      checked_add(linear_alpha_width, linear_gate_width,
                  "linear pack width overflow"),
      "linear pack width overflow");
  const std::size_t full_q_width = checked_mul(
      config.full_attention_head_count, config.full_attention_head_dimension,
      "full query width overflow");
  const std::size_t full_kv_width = checked_mul(
      config.full_attention_kv_head_count, config.full_attention_head_dimension,
      "full KV width overflow");
  const std::size_t full_kv_cache_bytes = attention_cache_bytes(
      kernels::Qwen35AttentionShape{
          .tokens = 1,
          .query_heads = config.full_attention_head_count,
          .kv_heads = config.full_attention_kv_head_count,
          .head_dim = config.full_attention_head_dimension,
          .max_context_tokens = max_context,
          .past_tokens = 0,
      },
      attention_policy);

  std::size_t bytes = 0;
  auto add = [&](std::size_t value) {
    bytes = checked_add(align_up(bytes, Qwen35Executor::workspace_alignment),
                        align_up(value, Qwen35Executor::workspace_alignment),
                        "Qwen3.5 executor workspace overflow");
  };

  add(checked_mul(max_context, sizeof(std::int32_t),
                  "token buffer byte size overflow"));
  add(sizeof(std::int32_t));
  add(sizeof(std::int32_t));
  add(sizeof(std::uint32_t));
  add(checked_mul(max_context, sizeof(std::int32_t),
                  "decode result buffer byte size overflow"));
  add(checked_mul(max_tokens, hidden * element,
                  "hidden activation byte size overflow"));
  add(checked_mul(max_tokens, hidden * element,
                  "hidden scratch byte size overflow"));
  add(checked_mul(max_tokens, intermediate * element,
                  "intermediate activation byte size overflow"));
  add(checked_mul(max_tokens, intermediate * element,
                  "intermediate scratch byte size overflow"));
  add(checked_mul(max_tokens, linear_pack_width * element,
                  "linear packed input byte size overflow"));
  add(checked_mul(max_tokens, qkv_width * element,
                  "linear qkv byte size overflow"));
  add(checked_mul(max_tokens, linear_beta_width * element,
                  "linear beta byte size overflow"));
  add(checked_mul(max_tokens, linear_alpha_width * element,
                  "linear alpha byte size overflow"));
  add(checked_mul(max_tokens, linear_gate_width * element,
                  "linear gate byte size overflow"));
  add(checked_mul(max_tokens, full_q_width * 2 * element,
                  "full query gate byte size overflow"));
  add(checked_mul(max_tokens, full_q_width * element,
                  "full query byte size overflow"));
  add(checked_mul(max_tokens, full_q_width * element,
                  "full query norm byte size overflow"));
  add(checked_mul(max_tokens, full_kv_width * element,
                  "full key byte size overflow"));
  add(checked_mul(max_tokens, full_kv_width * element,
                  "full key norm byte size overflow"));
  add(checked_mul(max_tokens, full_kv_width * element,
                  "full value byte size overflow"));
  add(checked_mul(max_tokens, hidden * element,
                  "attention output byte size overflow"));
  add(checked_mul(max_tokens, hidden * element,
                  "mixer projection byte size overflow"));
  add(checked_mul(config.vocabulary_size * element, std::size_t{1},
                  "logits byte size overflow"));
  add(checked_mul(checked_mul(max_tokens, std::max(hidden, intermediate),
                              "matmul input conversion element count overflow"),
                  std::size_t{2},
                  "matmul input conversion byte size overflow"));
  add(kMatmulWorkspaceBudget);
  add(max_attention_workspace(config, max_tokens, max_context,
                              attention_policy));
  const kernels::GatedDeltaShape delta_shape{
      .tokens = max_tokens,
      .hidden_size = hidden,
      .key_heads = config.linear_key_head_count,
      .value_heads = config.linear_value_head_count,
      .key_dim = config.linear_head_dimension,
      .value_dim = config.linear_head_dimension,
      .conv_width = config.linear_convolution_width,
      .epsilon = config.rms_norm_epsilon,
  };
  add(kernels::qwen35_gated_delta_workspace_bytes(delta_shape));

  for (const auto &block : config.blocks) {
    if (block.kind == model::Qwen35BlockKind::full_attention) {
      add(full_kv_cache_bytes);
    } else {
      add(checked_mul(config.linear_convolution_width - 1,
                      qkv_width * sizeof(float),
                      "linear convolution state byte size overflow"));
      const std::size_t recurrent =
          checked_mul(config.linear_value_head_count,
                      checked_mul(config.linear_head_dimension,
                                  config.linear_head_dimension,
                                  "linear recurrent state byte size overflow"),
                      "linear recurrent state byte size overflow");
      add(checked_mul(recurrent, sizeof(float),
                      "linear recurrent state byte size overflow"));
    }
  }
  return align_up(bytes, Qwen35Executor::workspace_alignment);
}

} // namespace

#if defined(BRT_QWEN35_EXECUTOR_TESTING)
namespace test {

struct InputCastLaunches {
  std::size_t total{};
  std::vector<std::size_t> full_attention_projection;
  std::vector<std::size_t> linear_attention_projection;
  std::vector<std::size_t> ffn_gate_up;
};

namespace {

InputCastLaunches input_cast_launches;
std::size_t *active_input_cast_group{};

} // namespace

void reset_qwen35_executor_input_cast_launches() {
  input_cast_launches = {};
  active_input_cast_group = nullptr;
}

InputCastLaunches qwen35_executor_input_cast_launches() {
  return input_cast_launches;
}

} // namespace test

namespace {

class InputCastGroupCounter {
public:
  explicit InputCastGroupCounter(std::vector<std::size_t> &groups)
      : previous_(test::active_input_cast_group) {
    groups.push_back(0);
    test::active_input_cast_group = &groups.back();
  }

  ~InputCastGroupCounter() { test::active_input_cast_group = previous_; }

  InputCastGroupCounter(const InputCastGroupCounter &) = delete;
  InputCastGroupCounter &operator=(const InputCastGroupCounter &) = delete;

private:
  std::size_t *previous_{};
};

void record_input_cast_launch() {
  ++test::input_cast_launches.total;
  if (test::active_input_cast_group != nullptr)
    ++*test::active_input_cast_group;
}

} // namespace
#endif

class Qwen35Executor::Impl {
public:
  Impl(ExecutionContext &context, const model::Qwen35Config &config,
       const model::CudaWeightPlan &weights, std::size_t max_context,
       Qwen35ExecutionPolicy policy)
      : context_(context), config_(config), weights_(weights),
        max_context_(max_context),
        policy_(resolve_execution_policy(config, max_context, policy)),
        dtype_(BRT_DTYPE_F32),
        weight_dtype_(dtype_from_weight(weights.token_embedding().type)),
        element_bytes_(dtype_size(dtype_)) {
    DeviceGuard guard{context_.device_id()};
    require(weights_.layer_count() == config_.blocks.size(),
            "CUDA weight layer count does not match Qwen3.5 config");
    validate_weight_dtypes(weights_);
    allocate_buffers();
    create_plans();
    create_delta_schedule_diagnostics();
    reset();
  }

  ~Impl() noexcept = default;

  Qwen35ExecutorResult prefill(std::span<const std::int32_t> tokens) {
    DeviceGuard guard{context_.device_id()};
    validate_request(config_, position_, tokens);
    ensure_healthy();
    std::size_t offset = 0;
    Qwen35ExecutorResult result{};
    const std::size_t start_position = position_;
    try {
      while (offset < tokens.size()) {
        const std::size_t remaining = tokens.size() - offset;
        const std::size_t chunk = remaining >= 512   ? 512
                                  : remaining >= 128 ? 128
                                  : remaining >= 17  ? 17
                                  : remaining >= 4   ? 4
                                  : remaining >= 2   ? 2
                                                     : 1;
        result =
            run_chunk(tokens.subspan(offset, chunk),
                      offset + chunk == tokens.size(), start_position + offset);
        offset += chunk;
      }
      position_ = start_position + tokens.size();
      sync_device_decode_position();
      return result;
    } catch (...) {
      poisoned_ = true;
      throw;
    }
  }

  Qwen35ExecutorResult decode(std::int32_t token) {
    DeviceGuard guard{context_.device_id()};
    const std::array<std::int32_t, 1> one{token};
    validate_request(config_, position_, one);
    ensure_healthy();
    const std::size_t start_position = position_;
    try {
      if (can_replay_decode_graph()) {
        upload_decode_token(token);
        decode_graph_->replay_on_current_device();
        check_cuda(cudaMemcpyAsync(host_decode_result_.get(), device_result_,
                                   sizeof(*host_decode_result_),
                                   cudaMemcpyDeviceToHost, context_.stream()),
                   "Qwen3.5 decode graph result download failed");
        check_cuda(cudaStreamSynchronize(context_.stream()),
                   "Qwen3.5 decode graph synchronization failed");
        decode_graph_replayed_ = true;
        position_ = start_position + 1;
        return Qwen35ExecutorResult{
            .token = *host_decode_result_,
            .position = static_cast<std::uint32_t>(start_position),
        };
      }
      auto result = run_chunk(one, true, start_position);
      position_ = start_position + 1;
      if (can_capture_decode_graph()) {
        capture_decode_graph();
      } else if (decode_graph_enabled() && decode_graph_ != nullptr &&
                 decode_graph_->captured()) {
        sync_device_decode_position();
      }
      return result;
    } catch (...) {
      poisoned_ = true;
      throw;
    }
  }

  Qwen35ExecutorResult decode_greedy(std::int32_t first_token,
                                     std::span<std::int32_t> output_tokens) {
    DeviceGuard guard{context_.device_id()};
    validate_decode_greedy_request(first_token, output_tokens.size());
    ensure_healthy();
    const std::size_t start_position = position_;
    try {
      if (!can_replay_decode_graph()) {
        const auto first = decode(first_token);
        output_tokens.front() = first.token;
        if (output_tokens.size() == 1) {
          return first;
        }
        if (!can_replay_decode_graph()) {
          return decode_greedy_sequential(first.token, output_tokens.subspan(1));
        }
        return replay_decode_greedy(first.token, output_tokens.subspan(1),
                                    start_position + 1);
      }
      return replay_decode_greedy(first_token, output_tokens, start_position);
    } catch (...) {
      poisoned_ = true;
      throw;
    }
  }

  void reset() {
    DeviceGuard guard{context_.device_id()};
    for (auto &state : full_states_) {
      check_cuda(cudaMemsetAsync(state.kv_cache, 0, state.kv_cache_bytes,
                                 context_.stream()),
                 "Qwen3.5 full-attention KV reset failed");
    }
    for (auto &state : linear_states_) {
      check_cuda(cudaMemsetAsync(state.convolution, 0, state.convolution_bytes,
                                 context_.stream()),
                 "Qwen3.5 linear convolution reset failed");
      check_cuda(cudaMemsetAsync(state.recurrent, 0, state.recurrent_bytes,
                                 context_.stream()),
                 "Qwen3.5 linear recurrent reset failed");
    }
    check_cuda(cudaMemsetAsync(device_decode_position_, 0,
                               sizeof(*device_decode_position_),
                               context_.stream()),
               "Qwen3.5 decode position reset failed");
    check_cuda(cudaStreamSynchronize(context_.stream()),
               "Qwen3.5 executor reset synchronization failed");
    position_ = 0;
    poisoned_ = false;
    decode_graph_replayed_ = false;
  }

  std::size_t position() const noexcept { return position_; }
  bool poisoned() const noexcept { return poisoned_; }
  Qwen35ExecutionDiagnostics diagnostics() const noexcept {
    return Qwen35ExecutionDiagnostics{
        .attention = policy_.attention,
        .kv_cache_dtype = policy_.kv_cache,
        .kv_cache_layout = policy_.kv_cache_layout,
        .decode_graph_captured =
            decode_graph_ != nullptr && decode_graph_->captured(),
        .decode_graph_replayed = decode_graph_replayed_,
        .attention_workspace_bytes = attention_workspace_bytes_,
        .cublaslt_algorithm_ids = cublaslt_algorithm_ids_,
        .cublaslt_plans = cublaslt_plan_diagnostics_,
        .gated_delta_schedules = gated_delta_schedule_diagnostics_,
    };
  }

  void copy_last_logits(std::span<float> output) const {
    DeviceGuard guard{context_.device_id()};
    require(output.size() == config_.vocabulary_size,
            "Qwen3.5 logits output size does not match vocabulary");
    require(position_ > 0, "Qwen3.5 logits are unavailable before execution");
    ensure_healthy();
    check_cuda(cudaMemcpyAsync(output.data(), logits_, output.size_bytes(),
                               cudaMemcpyDeviceToHost, context_.stream()),
               "Qwen3.5 logits download failed");
    check_cuda(cudaStreamSynchronize(context_.stream()),
               "Qwen3.5 logits synchronization failed");
  }

#if defined(BRT_QWEN35_EXECUTOR_TESTING)
  test::Qwen35ExecutorStateSnapshot state_snapshot_for_tests() const {
    DeviceGuard guard{context_.device_id()};
    test::Qwen35ExecutorStateSnapshot snapshot;
    const auto append = [&](std::vector<std::byte> &destination,
                            const void *source, std::size_t bytes) {
      const std::size_t offset = destination.size();
      destination.resize(offset + bytes);
      check_cuda(cudaMemcpyAsync(destination.data() + offset, source, bytes,
                                 cudaMemcpyDeviceToHost, context_.stream()),
                 "Qwen3.5 test state download failed");
    };
    for (const auto &state : full_states_)
      append(snapshot.full_kv_cache, state.kv_cache, state.kv_cache_bytes);
    for (const auto &state : linear_states_) {
      append(snapshot.linear_convolution, state.convolution,
             state.convolution_bytes);
      append(snapshot.linear_recurrent, state.recurrent, state.recurrent_bytes);
    }
    check_cuda(cudaStreamSynchronize(context_.stream()),
               "Qwen3.5 test state synchronization failed");
    return snapshot;
  }
#endif

  void enable_trace(bool enabled) {
    trace_enabled_ = enabled;
    trace_.clear();
  }

  const std::vector<Qwen35TraceEntry> &trace() const noexcept { return trace_; }

private:
  friend class Qwen35Executor;

  struct FullState {
    void *kv_cache{};
    std::size_t kv_cache_bytes{};
  };
  struct LinearState {
    float *convolution{};
    std::size_t convolution_bytes{};
    float *recurrent{};
    std::size_t recurrent_bytes{};
  };

  static void require_primary_dtype(model::CudaWeightType actual,
                                    model::CudaWeightType expected) {
    require(actual == expected,
            "Qwen3.5 CUDA executor primary weights must share the activation "
            "dtype");
  }

  static void require_auxiliary_dtype(model::CudaWeightType actual,
                                      model::CudaWeightType expected) {
    require(actual == expected || actual == model::CudaWeightType::f32,
            "Qwen3.5 CUDA executor auxiliary weights must match activations "
            "or be F32");
  }

  static void validate_weight_dtypes(const model::CudaWeightPlan &weights) {
    const auto expected = weights.token_embedding().type;
    require(expected == model::CudaWeightType::f16 ||
                expected == model::CudaWeightType::bf16,
            "Qwen3.5 CUDA executor activation dtype must be F16 or BF16");
    require_auxiliary_dtype(weights.output_norm().type, expected);
    require_primary_dtype(weights.output().type, expected);
    for (std::size_t layer = 0; layer < weights.layer_count(); ++layer) {
      const auto &w = weights.layer(layer);
      require_auxiliary_dtype(w.common.input_norm.type, expected);
      require_auxiliary_dtype(w.common.post_attention_norm.type, expected);
      require_primary_dtype(w.common.ffn_gate.type, expected);
      require_primary_dtype(w.common.ffn_down.type, expected);
      require_primary_dtype(w.common.ffn_up.type, expected);
      if (w.full_attention.has_value()) {
        const auto &full = *w.full_attention;
        require_primary_dtype(full.query.type, expected);
        require_primary_dtype(full.key.type, expected);
        require_primary_dtype(full.value.type, expected);
        require_primary_dtype(full.output.type, expected);
        require_auxiliary_dtype(full.query_norm.type, expected);
        require_auxiliary_dtype(full.key_norm.type, expected);
      }
      if (w.linear_attention.has_value()) {
        const auto &linear = *w.linear_attention;
        require_primary_dtype(linear.qkv.type, expected);
        require_primary_dtype(linear.gate.type, expected);
        require_auxiliary_dtype(linear.convolution.type, expected);
        require(linear.time_step_bias.type == linear.convolution.type &&
                    linear.recurrent_a.type == linear.convolution.type &&
                    linear.output_norm.type == linear.convolution.type,
                "Qwen3.5 gated-delta auxiliary weights must share one dtype");
        require_primary_dtype(linear.beta.type, expected);
        require_primary_dtype(linear.alpha.type, expected);
        require_primary_dtype(linear.output.type, expected);
      }
    }
  }

  std::size_t linear_convolution_state_tokens() const noexcept {
    return config_.linear_convolution_width > 0
               ? config_.linear_convolution_width - 1
               : 0;
  }

  void ensure_healthy() const {
    require(!poisoned_,
            "Qwen3.5 executor is poisoned; call reset before reuse");
  }

  bool decode_graph_enabled() const noexcept {
    return policy_.decode_graph &&
           policy_.attention == Qwen35AttentionImplementation::online_tiled;
  }

  bool can_replay_decode_graph() const noexcept {
    return !trace_enabled_ && decode_graph_enabled() &&
           decode_graph_ != nullptr && decode_graph_->captured();
  }

  bool can_capture_decode_graph() const noexcept {
    return !trace_enabled_ && decode_graph_enabled() &&
           (decode_graph_ == nullptr || !decode_graph_->captured());
  }

  void sync_device_decode_position() {
    require(position_ <= std::numeric_limits<std::uint32_t>::max(),
            "Qwen3.5 decode position exceeds CUDA graph range");
    const auto position = static_cast<std::uint32_t>(position_);
    check_cuda(cudaMemcpyAsync(device_decode_position_, &position,
                               sizeof(position), cudaMemcpyHostToDevice,
                               context_.stream()),
               "Qwen3.5 decode position upload failed");
    check_cuda(cudaStreamSynchronize(context_.stream()),
               "Qwen3.5 decode position synchronization failed");
  }

  void upload_decode_token(std::int32_t token) {
    *host_decode_token_ = token;
    check_cuda(cudaMemcpyAsync(device_decode_token_, host_decode_token_.get(),
                               sizeof(*host_decode_token_),
                               cudaMemcpyHostToDevice, context_.stream()),
               "Qwen3.5 decode graph token upload failed");
  }

  void validate_decode_greedy_request(std::int32_t first_token,
                                      std::size_t token_count) const {
    require(token_count != 0,
            "Qwen3.5 greedy decode output span must not be empty");
    require(token_count <= max_context_ - position_,
            "Qwen3.5 request exceeds session context");
    const std::array<std::int32_t, 1> one{first_token};
    validate_request(config_, position_, one);
  }

  Qwen35ExecutorResult replay_decode_greedy(
      std::int32_t first_token, std::span<std::int32_t> output_tokens,
      std::size_t start_position) {
    upload_decode_token(first_token);
    for (std::size_t step = 0; step < output_tokens.size(); ++step) {
      decode_graph_->replay_on_current_device();
    }
    check_cuda(cudaMemcpyAsync(output_tokens.data(),
                               device_decode_results_ + start_position,
                               output_tokens.size_bytes(),
                               cudaMemcpyDeviceToHost, context_.stream()),
               "Qwen3.5 greedy decode result download failed");
    check_cuda(cudaStreamSynchronize(context_.stream()),
               "Qwen3.5 greedy decode graph synchronization failed");
    decode_graph_replayed_ = true;
    position_ = start_position + output_tokens.size();
    return Qwen35ExecutorResult{
        .token = output_tokens.back(),
        .position = static_cast<std::uint32_t>(position_ - 1),
    };
  }

  Qwen35ExecutorResult
  decode_greedy_sequential(std::int32_t first_token,
                           std::span<std::int32_t> output_tokens) {
    auto token = first_token;
    Qwen35ExecutorResult result{};
    for (auto &output : output_tokens) {
      result = decode(token);
      output = result.token;
      token = result.token;
    }
    return result;
  }

  void capture_decode_graph() {
    sync_device_decode_position();
    if (decode_graph_ == nullptr) {
      decode_graph_ = std::make_unique<CudaGraphDecode>(context_.device_id(),
                                                        context_.stream());
    }
    decode_graph_->capture([this] {
      (void)run_chunk(
          std::span<const std::int32_t>{host_decode_token_.get(), 1}, true, 0,
          device_decode_token_, device_decode_position_,
          nullptr, false, device_result_);
      commit_decode_result<<<1, 1, 0, context_.stream()>>>(
          device_decode_token_, device_result_, device_decode_results_,
          device_decode_position_);
      check_cuda(cudaGetLastError(),
                 "Qwen3.5 decode graph result commit failed");
    });
  }

  void record_trace(std::string name, const void *device,
                    std::size_t elements) {
    if (!trace_enabled_)
      return;
    std::vector<float> host(elements);
    check_cuda(cudaMemcpyAsync(host.data(), device, host.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, context_.stream()),
               "Qwen3.5 trace download failed");
    check_cuda(cudaStreamSynchronize(context_.stream()),
               "Qwen3.5 trace synchronization failed");
    Qwen35TraceEntry entry{.name = std::move(name)};
    for (const float value : host)
      entry.sum += static_cast<double>(value);
    for (std::size_t index = 0; index < 3 && index < host.size(); ++index) {
      entry.first[index] = host[index];
      entry.last[2 - index] = host[host.size() - 1 - index];
    }
    trace_.push_back(std::move(entry));
  }

  void allocate_buffers() {
    auto &arena = context_.workspace();
    const auto attention_policy = attention_launch_policy(policy_);
    const std::size_t max_tokens =
        std::min<std::size_t>(kMaxPrefillTokens, max_context_);
    const std::size_t hidden = config_.hidden_size;
    const std::size_t intermediate = config_.intermediate_size;
    const std::size_t qkv_width = linear_qkv_width(config_);
    const std::size_t beta_width = config_.linear_value_head_count;
    const std::size_t alpha_width = config_.linear_value_head_count;
    const std::size_t linear_gate_width = checked_mul(
        config_.linear_value_head_count, config_.linear_head_dimension,
        "linear gate width overflow");
    const std::size_t linear_pack_width = checked_add(
        checked_add(qkv_width, beta_width, "linear pack width overflow"),
        checked_add(alpha_width, linear_gate_width,
                    "linear pack width overflow"),
        "linear pack width overflow");
    const std::size_t full_q_width = checked_mul(
        config_.full_attention_head_count,
        config_.full_attention_head_dimension, "full query width overflow");
    const std::size_t full_kv_width = checked_mul(
        config_.full_attention_kv_head_count,
        config_.full_attention_head_dimension, "full KV width overflow");
    const auto activation_bytes = [&](std::size_t tokens, std::size_t width,
                                      const char *message) {
      return checked_mul(checked_mul(tokens, width, message), element_bytes_,
                         message);
    };

    device_tokens_ = static_cast<std::int32_t *>(
        allocate(arena,
                 checked_mul(max_context_, sizeof(std::int32_t),
                             "token buffer byte size overflow"),
                 alignof(std::int32_t)));
    device_result_ = static_cast<std::int32_t *>(
        allocate(arena, sizeof(std::int32_t), alignof(std::int32_t)));
    device_decode_token_ = static_cast<std::int32_t *>(
        allocate(arena, sizeof(std::int32_t), alignof(std::int32_t)));
    device_decode_position_ = static_cast<std::uint32_t *>(
        allocate(arena, sizeof(std::uint32_t), alignof(std::uint32_t)));
    device_decode_results_ = static_cast<std::int32_t *>(
        allocate(arena,
                 checked_mul(max_context_, sizeof(std::int32_t),
                             "decode result buffer byte size overflow"),
                 alignof(std::int32_t)));
    std::int32_t *host_decode_token{};
    check_cuda(cudaHostAlloc(&host_decode_token, sizeof(*host_decode_token),
                             cudaHostAllocDefault),
               "Qwen3.5 decode token pinning failed");
    host_decode_token_.reset(host_decode_token);
    std::int32_t *host_decode_result{};
    check_cuda(cudaHostAlloc(&host_decode_result, sizeof(*host_decode_result),
                             cudaHostAllocDefault),
               "Qwen3.5 decode result pinning failed");
    host_decode_result_.reset(host_decode_result);
    hidden_a_ =
        allocate(arena,
                 activation_bytes(max_tokens, hidden,
                                  "hidden activation byte size overflow"),
                 16);
    hidden_b_ = allocate(arena,
                         activation_bytes(max_tokens, hidden,
                                          "hidden scratch byte size overflow"),
                         16);
    intermediate_a_ =
        allocate(arena,
                 activation_bytes(max_tokens, intermediate,
                                  "intermediate activation byte size overflow"),
                 16);
    intermediate_b_ =
        allocate(arena,
                 activation_bytes(max_tokens, intermediate,
                                  "intermediate scratch byte size overflow"),
                 16);
    linear_pack_ =
        allocate(arena,
                 activation_bytes(max_tokens, linear_pack_width,
                                  "linear packed input byte size overflow"),
                 16);
    linear_qkv_ = allocate(arena,
                           activation_bytes(max_tokens, qkv_width,
                                            "linear qkv byte size overflow"),
                           16);
    linear_beta_ = allocate(arena,
                            activation_bytes(max_tokens, beta_width,
                                             "linear beta byte size overflow"),
                            16);
    linear_alpha_ =
        allocate(arena,
                 activation_bytes(max_tokens, alpha_width,
                                  "linear alpha byte size overflow"),
                 16);
    linear_gate_ = allocate(arena,
                            activation_bytes(max_tokens, linear_gate_width,
                                             "linear gate byte size overflow"),
                            16);
    full_query_gate_ =
        allocate(arena,
                 activation_bytes(max_tokens,
                                  checked_mul(full_q_width, 2,
                                              "full query gate width overflow"),
                                  "full query gate byte size overflow"),
                 16);
    full_query_ = allocate(arena,
                           activation_bytes(max_tokens, full_q_width,
                                            "full query byte size overflow"),
                           16);
    full_query_norm_ =
        allocate(arena,
                 activation_bytes(max_tokens, full_q_width,
                                  "full query norm byte size overflow"),
                 16);
    full_key_ = allocate(arena,
                         activation_bytes(max_tokens, full_kv_width,
                                          "full key byte size overflow"),
                         16);
    full_key_norm_ =
        allocate(arena,
                 activation_bytes(max_tokens, full_kv_width,
                                  "full key norm byte size overflow"),
                 16);
    full_value_ = allocate(arena,
                           activation_bytes(max_tokens, full_kv_width,
                                            "full value byte size overflow"),
                           16);
    attention_out_ =
        allocate(arena,
                 activation_bytes(max_tokens, hidden,
                                  "attention output byte size overflow"),
                 16);
    mixer_projected_ =
        allocate(arena,
                 activation_bytes(max_tokens, hidden,
                                  "mixer projection byte size overflow"),
                 16);
    logits_ = allocate(arena,
                       checked_mul(config_.vocabulary_size, element_bytes_,
                                   "logits byte size overflow"),
                       16);
    matmul_input_ = allocate(
        arena,
        checked_mul(
            checked_mul(max_tokens, std::max(hidden, intermediate),
                        "matmul input conversion element count overflow"),
            dtype_size(weight_dtype_),
            "matmul input conversion byte size overflow"),
        16);
    matmul_workspace_ = allocate(arena, kMatmulWorkspaceBudget,
                                 Qwen35Executor::workspace_alignment);
    attention_workspace_bytes_ = max_attention_workspace(
        config_, max_tokens, max_context_, attention_policy);
    if (attention_workspace_bytes_ != 0) {
      attention_workspace_ =
          allocate(arena, attention_workspace_bytes_, alignof(float));
    }
    const kernels::GatedDeltaShape delta_shape{
        .tokens = max_tokens,
        .hidden_size = hidden,
        .key_heads = config_.linear_key_head_count,
        .value_heads = config_.linear_value_head_count,
        .key_dim = config_.linear_head_dimension,
        .value_dim = config_.linear_head_dimension,
        .conv_width = config_.linear_convolution_width,
        .epsilon = config_.rms_norm_epsilon,
    };
    delta_workspace_ = allocate(
        arena, kernels::qwen35_gated_delta_workspace_bytes(delta_shape),
        alignof(float));

    full_states_.reserve(config_.blocks.size());
    linear_states_.reserve(config_.blocks.size());
    full_state_by_layer_.assign(config_.blocks.size(),
                                std::numeric_limits<std::size_t>::max());
    linear_state_by_layer_.assign(config_.blocks.size(),
                                  std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 0; i < config_.blocks.size(); ++i) {
      if (config_.blocks[i].kind == model::Qwen35BlockKind::full_attention) {
        full_state_by_layer_[i] = full_states_.size();
        const std::size_t kv_cache_bytes = attention_cache_bytes(
            kernels::Qwen35AttentionShape{
                .tokens = 1,
                .query_heads = config_.full_attention_head_count,
                .kv_heads = config_.full_attention_kv_head_count,
                .head_dim = config_.full_attention_head_dimension,
                .max_context_tokens = max_context_,
                .past_tokens = 0,
            },
            attention_policy);
        full_states_.push_back(FullState{
            .kv_cache = allocate(arena, kv_cache_bytes, alignof(float)),
            .kv_cache_bytes = kv_cache_bytes});
      } else {
        linear_state_by_layer_[i] = linear_states_.size();
        const std::size_t recurrent = checked_mul(
            config_.linear_value_head_count,
            checked_mul(config_.linear_head_dimension,
                        config_.linear_head_dimension,
                        "linear recurrent state element count overflow"),
            "linear recurrent state element count overflow");
        const std::size_t convolution_bytes = checked_mul(
            checked_mul(linear_convolution_state_tokens(), qkv_width,
                        "linear convolution state element count overflow"),
            sizeof(float), "linear convolution state byte size overflow");
        const std::size_t recurrent_bytes =
            checked_mul(recurrent, sizeof(float),
                        "linear recurrent state byte size overflow");
        linear_states_.push_back(LinearState{
            .convolution = static_cast<float *>(
                allocate(arena, convolution_bytes, alignof(float))),
            .convolution_bytes = convolution_bytes,
            .recurrent = static_cast<float *>(
                allocate(arena, recurrent_bytes, alignof(float))),
            .recurrent_bytes = recurrent_bytes,
        });
      }
    }
  }

  void create_plans() {
    std::vector<PlanCacheEntry> plan_cache;
    const auto create_plan = [&](const CublasLtMatmulConfig &config,
                                 const model::CudaTensorView &weight,
                                 void *output, std::size_t bucket) {
      for (const auto &entry : plan_cache) {
        if (same_matmul_config(entry.config, config))
          return entry.plan;
      }
      auto unique = CublasLtMatmulPlan::create(config);
      std::shared_ptr<CublasLtMatmulPlan> plan{std::move(unique)};
      const bool tuned =
          release_tuning_bucket(bucket) && bucket <= max_context_;
      if (tuned) {
        check_cuda(cudaMemsetAsync(matmul_input_, 0, plan->input_bytes(),
                                   context_.stream()),
                   "Qwen3.5 cuBLASLt tuning input initialization failed");
        plan->select_fastest(context_.stream(), matmul_input_,
                             weight.device_data, output, matmul_workspace_,
                             kMatmulWorkspaceBudget);
      }
      cublaslt_algorithm_ids_.push_back(plan->algorithm_id());
      cublaslt_plan_diagnostics_.push_back(Qwen35CublasLtPlanDiagnostic{
          .bucket_tokens = bucket,
          .m = config.shape.m,
          .n = config.shape.n,
          .k = config.shape.k,
          .tuned = tuned,
          .algorithm_id = plan->algorithm_id(),
          .workspace_bytes = plan->workspace_bytes(),
      });
      plan_cache.push_back(PlanCacheEntry{.config = config, .plan = plan});
      return plan;
    };

    bucket_plans_.reserve(kBuckets.size());
    for (const std::size_t bucket : kBuckets) {
      auto &entry = bucket_plans_.emplace_back();
      entry.tokens = bucket;
      entry.layers.reserve(config_.blocks.size());
      for (std::size_t layer = 0; layer < config_.blocks.size(); ++layer) {
        const auto &block = config_.blocks[layer];
        const auto &weights = weights_.layer(layer);
        require(weights.index == block.index,
                "Qwen3.5 CUDA layer index does not match config");
        auto &plans = entry.layers.emplace_back();
        plans.ffn_gate = create_plan(
            matmul_config(bucket, config_.intermediate_size,
                          config_.hidden_size, dtype_, weight_dtype_),
            weights.common.ffn_gate, intermediate_a_, bucket);
        plans.ffn_up = create_plan(
            matmul_config(bucket, config_.intermediate_size,
                          config_.hidden_size, dtype_, weight_dtype_),
            weights.common.ffn_up, intermediate_b_, bucket);
        plans.ffn_down = create_plan(
            matmul_config(bucket, config_.hidden_size,
                          config_.intermediate_size, dtype_, weight_dtype_),
            weights.common.ffn_down, mixer_projected_, bucket);
        if (block.kind == model::Qwen35BlockKind::full_attention) {
          require(weights.full_attention.has_value(),
                  "full-attention block is missing CUDA weights");
          const std::size_t q_width = config_.full_attention_head_count *
                                      config_.full_attention_head_dimension;
          const std::size_t kv_width = config_.full_attention_kv_head_count *
                                       config_.full_attention_head_dimension;
          plans.full = std::make_unique<FullPlans>();
          plans.full->query_gate = create_plan(
              matmul_config(bucket, q_width * 2, config_.hidden_size, dtype_,
                            weight_dtype_),
              weights.full_attention->query, full_query_gate_, bucket);
          plans.full->key =
              create_plan(matmul_config(bucket, kv_width, config_.hidden_size,
                                        dtype_, weight_dtype_),
                          weights.full_attention->key, full_key_, bucket);
          plans.full->value =
              create_plan(matmul_config(bucket, kv_width, config_.hidden_size,
                                        dtype_, weight_dtype_),
                          weights.full_attention->value, full_value_, bucket);
          plans.full->output = create_plan(
              matmul_config(bucket, config_.hidden_size, q_width, dtype_,
                            weight_dtype_),
              weights.full_attention->output, mixer_projected_, bucket);
        } else {
          require(weights.linear_attention.has_value(),
                  "linear-attention block is missing CUDA weights");
          const std::size_t qkv_width = linear_qkv_width(config_);
          const std::size_t linear_gate_width =
              config_.linear_value_head_count * config_.linear_head_dimension;
          plans.linear = std::make_unique<LinearPlans>();
          plans.linear->qkv =
              create_plan(matmul_config(bucket, qkv_width, config_.hidden_size,
                                        dtype_, weight_dtype_),
                          weights.linear_attention->qkv, linear_qkv_, bucket);
          plans.linear->beta = create_plan(
              matmul_config(bucket, config_.linear_value_head_count,
                            config_.hidden_size, dtype_, weight_dtype_),
              weights.linear_attention->beta, linear_beta_, bucket);
          plans.linear->alpha = create_plan(
              matmul_config(bucket, config_.linear_value_head_count,
                            config_.hidden_size, dtype_, weight_dtype_),
              weights.linear_attention->alpha, linear_alpha_, bucket);
          plans.linear->gate = create_plan(
              matmul_config(bucket, linear_gate_width, config_.hidden_size,
                            dtype_, weight_dtype_),
              weights.linear_attention->gate, linear_gate_, bucket);
          plans.linear->output = create_plan(
              matmul_config(bucket, config_.hidden_size, linear_gate_width,
                            dtype_, weight_dtype_),
              weights.linear_attention->output, mixer_projected_, bucket);
        }
      }
      entry.lm_head =
          create_plan(matmul_config(1, config_.vocabulary_size,
                                    config_.hidden_size, dtype_, weight_dtype_),
                      weights_.output(), logits_, 1);
    }
  }

  void create_delta_schedule_diagnostics() {
    const kernels::GatedDeltaShape shape{
        .tokens = 1,
        .hidden_size = config_.hidden_size,
        .key_heads = config_.linear_key_head_count,
        .value_heads = config_.linear_value_head_count,
        .key_dim = config_.linear_head_dimension,
        .value_dim = config_.linear_head_dimension,
        .conv_width = config_.linear_convolution_width,
        .epsilon = config_.rms_norm_epsilon,
    };
    const model::Qwen35CudaLinearAttentionWeights *linear_weights = nullptr;
    for (std::size_t layer = 0; layer < weights_.layer_count(); ++layer) {
      const auto &weights = weights_.layer(layer);
      if (weights.linear_attention.has_value()) {
        linear_weights = &*weights.linear_attention;
        break;
      }
    }
    for (const std::size_t bucket :
         {std::size_t{1}, std::size_t{128}, std::size_t{512}}) {
      if (bucket <= max_context_) {
        if (linear_weights == nullptr || linear_states_.empty()) {
          auto diagnostic =
              kernels::qwen35_gated_delta_schedule_diagnostic(shape, bucket);
          diagnostic.rejection_reason = "no_linear_attention_layer";
          gated_delta_schedule_diagnostics_.push_back(std::move(diagnostic));
        } else {
          gated_delta_schedule_diagnostics_.push_back(
              tune_delta_schedule(shape, bucket, *linear_weights));
        }
      }
    }
  }

  void reset_delta_tuning_buffers(const kernels::GatedDeltaShape &shape,
                                  std::size_t tokens) {
    const std::size_t key_width = checked_mul(
        shape.key_heads, shape.key_dim, "delta tuning key width overflow");
    const std::size_t value_width =
        checked_mul(shape.value_heads, shape.value_dim,
                    "delta tuning value width overflow");
    const std::size_t qkv_width =
        checked_add(checked_mul(key_width, std::size_t{2},
                                "delta tuning qkv width overflow"),
                    value_width, "delta tuning qkv width overflow");
    const std::size_t scalar_width =
        checked_mul(shape.value_heads, std::size_t{2},
                    "delta tuning scalar width overflow");
    const std::size_t packed_width =
        checked_add(checked_add(qkv_width, scalar_width,
                                "delta tuning packed width overflow"),
                    shape.hidden_size, "delta tuning packed width overflow");
    const std::size_t input_elements =
        checked_mul(tokens, packed_width, "delta tuning input overflow");
    const int block = 256;
    const std::size_t grid_size =
        (input_elements + static_cast<std::size_t>(block) - 1) /
        static_cast<std::size_t>(block);
    require(grid_size <=
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "delta tuning input fill grid overflow");
    const int grid = static_cast<int>(grid_size);
    fill_delta_tuning_input<<<grid, block, 0, context_.stream()>>>(
        linear_pack_, input_elements, dtype_);
    check_cuda(cudaGetLastError(), "Qwen3.5 delta tuning input fill failed");
    auto &state = linear_states_.front();
    check_cuda(cudaMemsetAsync(state.convolution, 0, state.convolution_bytes,
                               context_.stream()),
               "Qwen3.5 delta tuning convolution reset failed");
    check_cuda(cudaMemsetAsync(state.recurrent, 0, state.recurrent_bytes,
                               context_.stream()),
               "Qwen3.5 delta tuning recurrent reset failed");
  }

  void
  run_delta_tuning_once(const kernels::GatedDeltaShape &shape,
                        const model::Qwen35CudaLinearAttentionWeights &linear,
                        kernels::GatedDeltaLaunchPolicy policy, void *output) {
    auto &state = linear_states_.front();
    kernels::qwen35_gated_delta(
        linear_pack_, linear.convolution.device_data,
        linear.recurrent_a.device_data, linear.time_step_bias.device_data,
        linear.output_norm.device_data, output, state.convolution,
        state.recurrent, delta_workspace_,
        kernels::qwen35_gated_delta_workspace_bytes(shape), shape, policy,
        dtype_, dtype_from_weight(linear.convolution.type), context_.stream());
  }

  std::vector<float> download_activation_f32(const void *device,
                                             std::size_t elements,
                                             const char *message) {
    std::vector<float> host(elements);
    if (dtype_ == BRT_DTYPE_F32) {
      check_cuda(cudaMemcpyAsync(host.data(), device, elements * sizeof(float),
                                 cudaMemcpyDeviceToHost, context_.stream()),
                 message);
    } else if (dtype_ == BRT_DTYPE_F16) {
      std::vector<__half> raw(elements);
      check_cuda(cudaMemcpyAsync(raw.data(), device, elements * sizeof(__half),
                                 cudaMemcpyDeviceToHost, context_.stream()),
                 message);
      check_cuda(cudaStreamSynchronize(context_.stream()), message);
      for (std::size_t index = 0; index < elements; ++index)
        host[index] = __half2float(raw[index]);
      return host;
    } else if (dtype_ == BRT_DTYPE_BF16) {
      std::vector<__nv_bfloat16> raw(elements);
      check_cuda(cudaMemcpyAsync(raw.data(), device,
                                 elements * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToHost, context_.stream()),
                 message);
      check_cuda(cudaStreamSynchronize(context_.stream()), message);
      for (std::size_t index = 0; index < elements; ++index)
        host[index] = __bfloat162float(raw[index]);
      return host;
    } else {
      throw Qwen35ExecutorError("unsupported Qwen3.5 delta tuning dtype");
    }
    check_cuda(cudaStreamSynchronize(context_.stream()), message);
    return host;
  }

  std::vector<float> download_state_f32(const float *device,
                                        std::size_t elements,
                                        const char *message) {
    std::vector<float> host(elements);
    check_cuda(cudaMemcpyAsync(host.data(), device, elements * sizeof(float),
                               cudaMemcpyDeviceToHost, context_.stream()),
               message);
    check_cuda(cudaStreamSynchronize(context_.stream()), message);
    return host;
  }

  bool compare_f32(std::span<const float> lhs, std::span<const float> rhs) {
    if (lhs.size() != rhs.size())
      return false;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
      if (!close_enough(lhs[index], rhs[index]))
        return false;
    }
    return true;
  }

  bool
  delta_candidate_correct(const kernels::GatedDeltaShape &shape,
                          const model::Qwen35CudaLinearAttentionWeights &linear,
                          kernels::GatedDeltaLaunchPolicy candidate) {
    constexpr kernels::GatedDeltaLaunchPolicy current{};
    reset_delta_tuning_buffers(shape, shape.tokens);
    run_delta_tuning_once(shape, linear, current, attention_out_);
    auto &state = linear_states_.front();
    const auto current_output = download_activation_f32(
        attention_out_, shape.tokens * shape.hidden_size,
        "Qwen3.5 delta tuning output download failed");
    const auto current_convolution = download_state_f32(
        state.convolution, state.convolution_bytes / sizeof(float),
        "Qwen3.5 delta tuning convolution download failed");
    const auto current_recurrent = download_state_f32(
        state.recurrent, state.recurrent_bytes / sizeof(float),
        "Qwen3.5 delta tuning recurrent download failed");

    reset_delta_tuning_buffers(shape, shape.tokens);
    run_delta_tuning_once(shape, linear, candidate, mixer_projected_);
    const auto candidate_output = download_activation_f32(
        mixer_projected_, shape.tokens * shape.hidden_size,
        "Qwen3.5 delta tuning output download failed");
    const auto candidate_convolution = download_state_f32(
        state.convolution, state.convolution_bytes / sizeof(float),
        "Qwen3.5 delta tuning convolution download failed");
    const auto candidate_recurrent = download_state_f32(
        state.recurrent, state.recurrent_bytes / sizeof(float),
        "Qwen3.5 delta tuning recurrent download failed");

    return compare_f32(candidate_output, current_output) &&
           compare_f32(candidate_convolution, current_convolution) &&
           compare_f32(candidate_recurrent, current_recurrent);
  }

  float time_delta_policy(const kernels::GatedDeltaShape &shape,
                          const model::Qwen35CudaLinearAttentionWeights &linear,
                          kernels::GatedDeltaLaunchPolicy policy) {
    constexpr int kWarmups = 2;
    constexpr int kMeasurements = 5;
    CudaEvent start{"Qwen3.5 delta tuning start event creation failed"};
    CudaEvent stop{"Qwen3.5 delta tuning stop event creation failed"};
    std::vector<float> measurements;
    measurements.reserve(kMeasurements);
    for (int iteration = 0; iteration < kWarmups + kMeasurements; ++iteration) {
      reset_delta_tuning_buffers(shape, shape.tokens);
      check_cuda(cudaEventRecord(start.get(), context_.stream()),
                 "Qwen3.5 delta tuning start event record failed");
      run_delta_tuning_once(shape, linear, policy, attention_out_);
      check_cuda(cudaEventRecord(stop.get(), context_.stream()),
                 "Qwen3.5 delta tuning stop event record failed");
      check_cuda(cudaEventSynchronize(stop.get()),
                 "Qwen3.5 delta tuning event synchronization failed");
      if (iteration >= kWarmups) {
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()),
                   "Qwen3.5 delta tuning event timing failed");
        measurements.push_back(elapsed_ms);
      }
    }
    return median_ms(std::move(measurements));
  }

  kernels::GatedDeltaScheduleDiagnostic
  tune_delta_schedule(kernels::GatedDeltaShape shape, std::size_t bucket,
                      const model::Qwen35CudaLinearAttentionWeights &linear) {
    shape.tokens = bucket;
    auto diagnostic =
        kernels::qwen35_gated_delta_schedule_diagnostic(shape, bucket);
    const auto candidate_schedule =
        bucket == 1
            ? kernels::GatedDeltaSchedule::register_resident_decode_sm120
            : kernels::GatedDeltaSchedule::register_resident_prefill_sm120;
    diagnostic.candidate_schedule = candidate_schedule;
    const auto candidate = kernels::GatedDeltaLaunchPolicy{
        .schedule = candidate_schedule,
        .warps_per_block = 4,
        .transposed_boundary_state = false,
    };
    if (!((shape.key_dim == 64 || shape.key_dim == 128) &&
          shape.value_dim == shape.key_dim)) {
      diagnostic.rejection_reason = "unsupported_shape";
      return diagnostic;
    }

    diagnostic.correctness_passed =
        delta_candidate_correct(shape, linear, candidate);
    diagnostic.current_median_ms = time_delta_policy(shape, linear, {});
    if (diagnostic.correctness_passed) {
      diagnostic.candidate_median_ms =
          time_delta_policy(shape, linear, candidate);
    }
    if (!diagnostic.correctness_passed) {
      diagnostic.rejection_reason = "correctness_failed";
      return diagnostic;
    }
    constexpr float kMeaningfulSpeedup = 0.98F;
    if (diagnostic.candidate_median_ms <
        diagnostic.current_median_ms * kMeaningfulSpeedup) {
      diagnostic.schedule = candidate_schedule;
      diagnostic.candidate_accepted = true;
      diagnostic.rejection_reason.clear();
    } else {
      diagnostic.schedule =
          kernels::GatedDeltaSchedule::register_resident_current;
      diagnostic.candidate_accepted = false;
      diagnostic.rejection_reason = "candidate_not_faster";
    }
    return diagnostic;
  }

  kernels::GatedDeltaLaunchPolicy delta_policy_for_bucket(
      std::size_t tokens,
      const kernels::GatedDeltaShape &shape) const noexcept {
    for (const auto &diagnostic : gated_delta_schedule_diagnostics_) {
      if (diagnostic.bucket_tokens == tokens &&
          diagnostic.key_dim == shape.key_dim &&
          diagnostic.value_dim == shape.value_dim) {
        return kernels::GatedDeltaLaunchPolicy{
            .schedule = diagnostic.schedule,
            .warps_per_block = diagnostic.warps_per_block,
            .transposed_boundary_state = diagnostic.transposed_boundary_state,
        };
      }
    }
    return kernels::qwen35_gated_delta_select_policy(shape, tokens);
  }

  BucketPlans &bucket_for(std::size_t tokens) {
    for (auto &bucket : bucket_plans_) {
      if (bucket.tokens == tokens)
        return bucket;
    }
    throw Qwen35ExecutorError("unsupported Qwen3.5 executor token bucket");
  }

  struct MatmulBinding {
    const CublasLtMatmulPlan *plan;
    const model::CudaTensorView *weight;
    void *output;
  };

  void run_matmul_group(const void *f32_input,
                        std::span<const MatmulBinding> bindings) {
    require(f32_input != nullptr, "Qwen3.5 matmul input must not be null");
    require(!bindings.empty(), "Qwen3.5 matmul group must not be empty");
    require(bindings.front().plan != nullptr,
            "Qwen3.5 matmul group plan must not be null");
    const std::size_t input_bytes = bindings.front().plan->input_bytes();
    const std::size_t weight_element_bytes = dtype_size(weight_dtype_);
    require(input_bytes % weight_element_bytes == 0,
            "Qwen3.5 matmul input byte size is not dtype-aligned");
    for (const auto &binding : bindings) {
      require(binding.plan != nullptr && binding.weight != nullptr &&
                  binding.output != nullptr,
              "Qwen3.5 matmul group binding must not be null");
      require(binding.plan->input_bytes() == input_bytes,
              "Qwen3.5 matmul group inputs must have identical byte sizes");
      require(dtype_from_weight(binding.weight->type) == weight_dtype_,
              "Qwen3.5 matmul group inputs must have identical dtypes");
    }
    kernels::qwen35_cast_f32(static_cast<const float *>(f32_input),
                             matmul_input_, input_bytes / weight_element_bytes,
                             weight_dtype_, context_.stream());
#if defined(BRT_QWEN35_EXECUTOR_TESTING)
    record_input_cast_launch();
#endif
    for (const auto &binding : bindings) {
      binding.plan->run(context_.stream(), matmul_input_, input_bytes,
                        binding.weight->device_data, binding.weight->bytes,
                        binding.output, binding.plan->output_bytes(),
                        matmul_workspace_, kMatmulWorkspaceBudget);
    }
  }

  void run_matmul(const CublasLtMatmulPlan &plan, const void *input,
                  const model::CudaTensorView &weight, void *output) {
    const MatmulBinding binding{
        .plan = &plan, .weight = &weight, .output = output};
    run_matmul_group(input,
                     std::span<const MatmulBinding>{&binding, std::size_t{1}});
  }

  Qwen35ExecutorResult
  run_chunk(std::span<const std::int32_t> tokens, bool produce_logits,
            std::size_t chunk_position,
            const std::int32_t *fixed_device_token = nullptr,
            const std::uint32_t *device_position = nullptr,
            std::int32_t *fixed_host_result = nullptr,
            bool synchronize = true,
            std::int32_t *fixed_device_result = nullptr) {
    BucketPlans &bucket = bucket_for(tokens.size());
    std::int32_t *device_tokens =
        const_cast<std::int32_t *>(fixed_device_token);
    if (device_tokens == nullptr) {
      device_tokens = device_tokens_ + chunk_position;
      check_cuda(cudaMemcpyAsync(device_tokens, tokens.data(),
                                 tokens.size_bytes(), cudaMemcpyHostToDevice,
                                 context_.stream()),
                 "Qwen3.5 token upload failed");
    }
    kernels::qwen35_embedding(
        device_tokens, weights_.token_embedding().device_data, hidden_a_,
        kernels::EmbeddingShape{.tokens = tokens.size(),
                                .embedding_dim = config_.hidden_size,
                                .vocab_size = config_.vocabulary_size},
        dtype_, weight_dtype_, context_.stream());
    record_trace("model.input_embed", hidden_a_,
                 tokens.size() * config_.hidden_size);
    void *hidden = hidden_a_;
    void *scratch = hidden_b_;
    const std::size_t hidden_elements = tokens.size() * config_.hidden_size;

    for (std::size_t layer = 0; layer < config_.blocks.size(); ++layer) {
      const auto &block = config_.blocks[layer];
      const auto &weights = weights_.layer(layer);
      const auto &plans = bucket.layers[layer];
      kernels::qwen35_rms_norm(
          hidden, weights.common.input_norm.device_data, scratch,
          kernels::RmsNormShape{.rows = tokens.size(),
                                .cols = config_.hidden_size},
          config_.rms_norm_epsilon, dtype_,
          dtype_from_weight(weights.common.input_norm.type), context_.stream());
      record_trace("attn_norm-" + std::to_string(layer), scratch,
                   hidden_elements);
      if (block.kind == model::Qwen35BlockKind::full_attention) {
        run_full_layer(tokens.size(), scratch, mixer_projected_, weights,
                       *plans.full, chunk_position, device_position);
      } else {
        run_linear_layer(tokens.size(), scratch, mixer_projected_, weights,
                         *plans.linear, layer);
      }
      kernels::qwen35_residual_add(hidden, mixer_projected_, scratch,
                                   hidden_elements, dtype_, context_.stream());
      std::swap(hidden, scratch);
      record_trace("attn_residual-" + std::to_string(layer), hidden,
                   hidden_elements);

      kernels::qwen35_rms_norm(
          hidden, weights.common.post_attention_norm.device_data, scratch,
          kernels::RmsNormShape{.rows = tokens.size(),
                                .cols = config_.hidden_size},
          config_.rms_norm_epsilon, dtype_,
          dtype_from_weight(weights.common.post_attention_norm.type),
          context_.stream());
      record_trace("attn_post_norm-" + std::to_string(layer), scratch,
                   hidden_elements);
      {
#if defined(BRT_QWEN35_EXECUTOR_TESTING)
        InputCastGroupCounter ffn_casts{test::input_cast_launches.ffn_gate_up};
#endif
        if (policy_.grouped_input_casts) {
          const std::array bindings{
              MatmulBinding{.plan = plans.ffn_gate.get(),
                            .weight = &weights.common.ffn_gate,
                            .output = intermediate_a_},
              MatmulBinding{.plan = plans.ffn_up.get(),
                            .weight = &weights.common.ffn_up,
                            .output = intermediate_b_},
          };
          run_matmul_group(scratch, std::span<const MatmulBinding>{bindings});
        } else {
          run_matmul(*plans.ffn_gate, scratch, weights.common.ffn_gate,
                     intermediate_a_);
          run_matmul(*plans.ffn_up, scratch, weights.common.ffn_up,
                     intermediate_b_);
        }
      }
      kernels::qwen35_swiglu(intermediate_a_, intermediate_b_, intermediate_a_,
                             tokens.size() * config_.intermediate_size, dtype_,
                             context_.stream());
      record_trace("ffn_swiglu-" + std::to_string(layer), intermediate_a_,
                   tokens.size() * config_.intermediate_size);
      run_matmul(*plans.ffn_down, intermediate_a_, weights.common.ffn_down,
                 mixer_projected_);
      kernels::qwen35_residual_add(hidden, mixer_projected_, scratch,
                                   hidden_elements, dtype_, context_.stream());
      std::swap(hidden, scratch);
      record_trace("l_out-" + std::to_string(layer), hidden, hidden_elements);
    }

    if (!produce_logits) {
      return Qwen35ExecutorResult{
          .token = -1,
          .position =
              static_cast<std::uint32_t>(chunk_position + tokens.size() - 1),
      };
    }

    const std::size_t last_offset =
        (tokens.size() - 1) * config_.hidden_size * element_bytes_;
    kernels::qwen35_rms_norm(
        static_cast<const std::byte *>(hidden) + last_offset,
        weights_.output_norm().device_data, scratch,
        kernels::RmsNormShape{.rows = 1, .cols = config_.hidden_size},
        config_.rms_norm_epsilon, dtype_,
        dtype_from_weight(weights_.output_norm().type), context_.stream());
    run_matmul(*bucket.lm_head, scratch, weights_.output(), logits_);
    std::int32_t *device_result =
        fixed_device_result == nullptr ? device_result_ : fixed_device_result;
    kernels::qwen35_argmax_typed(logits_, device_result,
                                 config_.vocabulary_size, dtype_,
                                 context_.stream());
    if (fixed_host_result == nullptr && !synchronize) {
      return Qwen35ExecutorResult{
          .token = 0,
          .position =
              static_cast<std::uint32_t>(chunk_position + tokens.size() - 1),
      };
    }
    std::int32_t host_result = 0;
    std::int32_t *result =
        fixed_host_result == nullptr ? &host_result : fixed_host_result;
    require(synchronize || fixed_host_result != nullptr,
            "Qwen3.5 unsynchronized result download requires fixed host "
            "storage");
    check_cuda(cudaMemcpyAsync(result, device_result, sizeof(*result),
                               cudaMemcpyDeviceToHost, context_.stream()),
               "Qwen3.5 result download failed");
    if (synchronize) {
      check_cuda(cudaStreamSynchronize(context_.stream()),
                 "Qwen3.5 executor synchronization failed");
    }
    return Qwen35ExecutorResult{
        .token = synchronize ? *result : 0,
        .position =
            static_cast<std::uint32_t>(chunk_position + tokens.size() - 1),
    };
  }

  void run_full_layer(std::size_t tokens, const void *input, void *output,
                      const model::Qwen35CudaLayerWeights &weights,
                      const FullPlans &plans, std::size_t chunk_position,
                      const std::uint32_t *device_position = nullptr) {
    const auto &full = *weights.full_attention;
    const std::size_t kv_width = config_.full_attention_kv_head_count *
                                 config_.full_attention_head_dimension;
    {
#if defined(BRT_QWEN35_EXECUTOR_TESTING)
      InputCastGroupCounter projection_casts{
          test::input_cast_launches.full_attention_projection};
#endif
      if (policy_.grouped_input_casts) {
        const std::array bindings{
            MatmulBinding{.plan = plans.query_gate.get(),
                          .weight = &full.query,
                          .output = full_query_gate_},
            MatmulBinding{.plan = plans.key.get(),
                          .weight = &full.key,
                          .output = full_key_},
            MatmulBinding{.plan = plans.value.get(),
                          .weight = &full.value,
                          .output = full_value_},
        };
        run_matmul_group(input, std::span<const MatmulBinding>{bindings});
      } else {
        run_matmul(*plans.query_gate, input, full.query, full_query_gate_);
        run_matmul(*plans.key, input, full.key, full_key_);
        run_matmul(*plans.value, input, full.value, full_value_);
      }
    }
    kernels::qwen35_split_full_query_gate(
        full_query_gate_, full_query_, linear_gate_, tokens,
        config_.full_attention_head_count,
        config_.full_attention_head_dimension, dtype_, context_.stream());
    kernels::qwen35_qk_norm_rope(
        full_query_, full.query_norm.device_data, full_query_norm_,
        kernels::QkNormRopeShape{.tokens = tokens,
                                 .heads = config_.full_attention_head_count,
                                 .head_dim =
                                     config_.full_attention_head_dimension,
                                 .rotary_dim = config_.rotary_dimension,
                                 .position_offset = chunk_position,
                                 .rope_base = config_.rope_frequency_base},
        config_.rms_norm_epsilon, dtype_,
        dtype_from_weight(full.query_norm.type), context_.stream(),
        device_position);
    kernels::qwen35_qk_norm_rope(
        full_key_, full.key_norm.device_data, full_key_norm_,
        kernels::QkNormRopeShape{.tokens = tokens,
                                 .heads = config_.full_attention_kv_head_count,
                                 .head_dim =
                                     config_.full_attention_head_dimension,
                                 .rotary_dim = config_.rotary_dimension,
                                 .position_offset = chunk_position,
                                 .rope_base = config_.rope_frequency_base},
        config_.rms_norm_epsilon, dtype_, dtype_from_weight(full.key_norm.type),
        context_.stream(), device_position);
    const std::size_t state_index = full_state_by_layer_[weights.index];
    require(state_index < full_states_.size(),
            "missing full-attention executor state");
    const auto attention_shape = kernels::Qwen35AttentionShape{
        .tokens = tokens,
        .query_heads = config_.full_attention_head_count,
        .kv_heads = config_.full_attention_kv_head_count,
        .head_dim = config_.full_attention_head_dimension,
        .max_context_tokens = max_context_,
        .past_tokens = chunk_position,
    };
    if (device_position != nullptr &&
        policy_.attention == Qwen35AttentionImplementation::online_tiled) {
      kernels::qwen35_online_attention_decode(
          full_query_norm_, full_key_norm_, full_value_, linear_gate_,
          attention_out_, full_states_[state_index].kv_cache,
          full_states_[state_index].kv_cache_bytes, attention_shape, dtype_,
          policy_.kv_cache, policy_.kv_cache_layout, device_position,
          context_.stream());
    } else {
      kernels::qwen35_causal_attention(
          full_query_norm_, full_key_norm_, full_value_, linear_gate_,
          attention_out_, full_states_[state_index].kv_cache,
          full_states_[state_index].kv_cache_bytes, attention_workspace_,
          attention_workspace_bytes_, attention_shape, dtype_,
          attention_launch_policy(policy_), context_.stream());
    }
    (void)kv_width;
    run_matmul(*plans.output, attention_out_, full.output, output);
    record_trace("attn_out-" + std::to_string(weights.index), output,
                 tokens * config_.hidden_size);
  }

  void run_linear_layer(std::size_t tokens, const void *input, void *output,
                        const model::Qwen35CudaLayerWeights &weights,
                        const LinearPlans &plans, std::size_t layer) {
    const auto &linear = *weights.linear_attention;
    const std::size_t qkv_width = linear_qkv_width(config_);
    const std::size_t beta_width = config_.linear_value_head_count;
    const std::size_t alpha_width = config_.linear_value_head_count;
    const std::size_t gate_width =
        config_.linear_value_head_count * config_.linear_head_dimension;
    {
#if defined(BRT_QWEN35_EXECUTOR_TESTING)
      InputCastGroupCounter projection_casts{
          test::input_cast_launches.linear_attention_projection};
#endif
      if (policy_.grouped_input_casts) {
        const std::array bindings{
            MatmulBinding{.plan = plans.qkv.get(),
                          .weight = &linear.qkv,
                          .output = linear_qkv_},
            MatmulBinding{.plan = plans.beta.get(),
                          .weight = &linear.beta,
                          .output = linear_beta_},
            MatmulBinding{.plan = plans.alpha.get(),
                          .weight = &linear.alpha,
                          .output = linear_alpha_},
            MatmulBinding{.plan = plans.gate.get(),
                          .weight = &linear.gate,
                          .output = linear_gate_},
        };
        run_matmul_group(input, std::span<const MatmulBinding>{bindings});
      } else {
        run_matmul(*plans.qkv, input, linear.qkv, linear_qkv_);
        run_matmul(*plans.beta, input, linear.beta, linear_beta_);
        run_matmul(*plans.alpha, input, linear.alpha, linear_alpha_);
        run_matmul(*plans.gate, input, linear.gate, linear_gate_);
      }
    }
    record_trace("linear_attn_qkv_mixed-" + std::to_string(layer), linear_qkv_,
                 tokens * qkv_width);
    kernels::qwen35_pack_linear_delta_input(
        linear_qkv_, linear_beta_, linear_alpha_, linear_gate_, linear_pack_,
        tokens, qkv_width, beta_width, alpha_width, gate_width, dtype_,
        context_.stream());
    const std::size_t state_index = linear_state_by_layer_[layer];
    require(state_index < linear_states_.size(),
            "missing linear executor state");
    const auto delta_shape = kernels::GatedDeltaShape{
        .tokens = tokens,
        .hidden_size = config_.hidden_size,
        .key_heads = config_.linear_key_head_count,
        .value_heads = config_.linear_value_head_count,
        .key_dim = config_.linear_head_dimension,
        .value_dim = config_.linear_head_dimension,
        .conv_width = config_.linear_convolution_width,
        .epsilon = config_.rms_norm_epsilon,
    };
    kernels::qwen35_gated_delta(
        linear_pack_, linear.convolution.device_data,
        linear.recurrent_a.device_data, linear.time_step_bias.device_data,
        linear.output_norm.device_data, attention_out_,
        linear_states_[state_index].convolution,
        linear_states_[state_index].recurrent, delta_workspace_,
        kernels::qwen35_gated_delta_workspace_bytes(delta_shape), delta_shape,
        delta_policy_for_bucket(tokens, delta_shape), dtype_,
        dtype_from_weight(linear.convolution.type), context_.stream());
    record_trace("final_output-" + std::to_string(layer), attention_out_,
                 tokens * config_.hidden_size);
    run_matmul(*plans.output, attention_out_, linear.output, output);
    record_trace("linear_attn_out-" + std::to_string(layer), output,
                 tokens * config_.hidden_size);
  }

  ExecutionContext context_;
  const model::Qwen35Config &config_;
  const model::CudaWeightPlan &weights_;
  std::size_t max_context_{};
  Qwen35ExecutionPolicy policy_{};
  BrtDataType dtype_{};
  BrtDataType weight_dtype_{};
  std::size_t element_bytes_{};
  std::size_t position_{};
  bool poisoned_{};
  bool trace_enabled_{};

  std::int32_t *device_tokens_{};
  std::int32_t *device_result_{};
  std::int32_t *device_decode_token_{};
  std::uint32_t *device_decode_position_{};
  std::int32_t *device_decode_results_{};
  CudaHostInt32 host_decode_token_;
  CudaHostInt32 host_decode_result_;
  void *hidden_a_{};
  void *hidden_b_{};
  void *intermediate_a_{};
  void *intermediate_b_{};
  void *linear_pack_{};
  void *linear_qkv_{};
  void *linear_beta_{};
  void *linear_alpha_{};
  void *linear_gate_{};
  void *full_query_gate_{};
  void *full_query_{};
  void *full_query_norm_{};
  void *full_key_{};
  void *full_key_norm_{};
  void *full_value_{};
  void *attention_out_{};
  void *mixer_projected_{};
  void *logits_{};
  void *matmul_input_{};
  void *matmul_workspace_{};
  void *attention_workspace_{};
  std::size_t attention_workspace_bytes_{};
  void *delta_workspace_{};

  std::vector<FullState> full_states_;
  std::vector<LinearState> linear_states_;
  std::vector<std::size_t> full_state_by_layer_;
  std::vector<std::size_t> linear_state_by_layer_;
  std::vector<BucketPlans> bucket_plans_;
  std::vector<int> cublaslt_algorithm_ids_;
  std::vector<Qwen35CublasLtPlanDiagnostic> cublaslt_plan_diagnostics_;
  std::vector<kernels::GatedDeltaScheduleDiagnostic>
      gated_delta_schedule_diagnostics_;
  std::vector<Qwen35TraceEntry> trace_;
  std::unique_ptr<CudaGraphDecode> decode_graph_;
  bool decode_graph_replayed_{};
};

std::size_t Qwen35Executor::workspace_bytes(const model::Qwen35Config &config,
                                            std::size_t max_context,
                                            Qwen35ExecutionPolicy policy) {
  return workspace_estimate(config, max_context, policy);
}

void Qwen35Executor::validate_request(const model::Qwen35Config &config,
                                      std::size_t past_tokens,
                                      std::span<const std::int32_t> tokens) {
  require(!tokens.empty(), "Qwen3.5 executor token span must not be empty");
  require(config.vocabulary_size > 0, "vocabulary_size must be positive");
  require(config.context_length > 0, "context_length must be positive");
  require(past_tokens <= config.context_length,
          "past token count exceeds context length");
  require(tokens.size() <= config.context_length - past_tokens,
          "Qwen3.5 request exceeds context length");
  for (const std::int32_t token : tokens) {
    require(token >= 0, "Qwen3.5 token id is negative");
    require(static_cast<std::size_t>(token) < config.vocabulary_size,
            "Qwen3.5 token id is out of range");
  }
}

void Qwen35Executor::validate_weight_dtypes_for_tests(
    const model::CudaWeightPlan &weights) {
  Impl::validate_weight_dtypes(weights);
}

Qwen35Executor::Qwen35Executor(ExecutionContext &context,
                               const model::Qwen35Config &config,
                               const model::CudaWeightPlan &weights,
                               std::size_t max_context,
                               Qwen35ExecutionPolicy policy)
    : impl_(new Impl(context, config, weights, max_context, policy)) {}

Qwen35Executor::~Qwen35Executor() noexcept { delete impl_; }

Qwen35ExecutorResult
Qwen35Executor::prefill(std::span<const std::int32_t> tokens) {
  return impl_->prefill(tokens);
}

Qwen35ExecutorResult Qwen35Executor::decode(std::int32_t token) {
  return impl_->decode(token);
}

Qwen35ExecutorResult
Qwen35Executor::decode_greedy(std::int32_t first_token,
                              std::span<std::int32_t> output_tokens) {
  return impl_->decode_greedy(first_token, output_tokens);
}

void Qwen35Executor::copy_last_logits(std::span<float> output) const {
  impl_->copy_last_logits(output);
}

void Qwen35Executor::enable_trace(bool enabled) {
  impl_->enable_trace(enabled);
}

const std::vector<Qwen35TraceEntry> &Qwen35Executor::trace() const noexcept {
  return impl_->trace();
}

void Qwen35Executor::reset() { impl_->reset(); }

std::size_t Qwen35Executor::position() const noexcept {
  return impl_->position();
}

bool Qwen35Executor::poisoned() const noexcept { return impl_->poisoned(); }

Qwen35ExecutionDiagnostics Qwen35Executor::diagnostics() const noexcept {
  return impl_->diagnostics();
}

#if defined(BRT_QWEN35_EXECUTOR_TESTING)
test::Qwen35ExecutorStateSnapshot
Qwen35Executor::state_snapshot_for_tests() const {
  return impl_->state_snapshot_for_tests();
}
#endif

} // namespace brt
