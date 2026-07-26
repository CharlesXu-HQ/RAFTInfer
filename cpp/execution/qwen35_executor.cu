#include "qwen35_executor.hpp"

#include "cublaslt_matmul.hpp"

#include "../kernels/qwen35_attention.cuh"
#include "../kernels/qwen35_delta.cuh"
#include "../kernels/qwen35_primitives.cuh"

#include <brt/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
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

void require(bool condition, const char *message) {
  if (!condition)
    throw Qwen35ExecutorError(message);
}

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

void *allocate(WorkspaceArena &arena, std::size_t bytes,
               std::size_t alignment) {
  return arena.allocate(bytes == 0 ? 1 : align_up(bytes, alignment), alignment);
}

struct LinearPlans {
  std::unique_ptr<CublasLtMatmulPlan> qkv;
  std::unique_ptr<CublasLtMatmulPlan> beta;
  std::unique_ptr<CublasLtMatmulPlan> alpha;
  std::unique_ptr<CublasLtMatmulPlan> gate;
  std::unique_ptr<CublasLtMatmulPlan> output;
};

struct FullPlans {
  std::unique_ptr<CublasLtMatmulPlan> query_gate;
  std::unique_ptr<CublasLtMatmulPlan> key;
  std::unique_ptr<CublasLtMatmulPlan> value;
  std::unique_ptr<CublasLtMatmulPlan> output;
};

struct LayerPlans {
  std::unique_ptr<CublasLtMatmulPlan> ffn_gate;
  std::unique_ptr<CublasLtMatmulPlan> ffn_up;
  std::unique_ptr<CublasLtMatmulPlan> ffn_down;
  std::unique_ptr<LinearPlans> linear;
  std::unique_ptr<FullPlans> full;
};

struct BucketPlans {
  std::size_t tokens{};
  std::vector<LayerPlans> layers;
  std::unique_ptr<CublasLtMatmulPlan> lm_head;
};

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

std::size_t max_attention_workspace(const model::Qwen35Config &config,
                                    std::size_t tokens,
                                    std::size_t max_context) {
  if (config.full_attention_head_count == 0)
    return 0;
  return kernels::qwen35_attention_workspace_bytes(
      kernels::Qwen35AttentionShape{
          .tokens = tokens,
          .query_heads = config.full_attention_head_count,
          .kv_heads = config.full_attention_kv_head_count,
          .head_dim = config.full_attention_head_dimension,
          .max_context_tokens = max_context,
          .past_tokens = max_context - tokens});
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

std::size_t workspace_estimate(const model::Qwen35Config &config,
                               std::size_t max_context) {
  validate_config(config, max_context);
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
  const std::size_t full_kv_cache_bytes =
      checked_mul(checked_mul(max_context, std::size_t{2},
                              "full attention KV cache element count overflow"),
                  checked_mul(full_kv_width, sizeof(float),
                              "full attention KV cache byte size overflow"),
                  "full attention KV cache byte size overflow");

  std::size_t bytes = 0;
  auto add = [&](std::size_t value) {
    bytes = checked_add(align_up(bytes, Qwen35Executor::workspace_alignment),
                        align_up(value, Qwen35Executor::workspace_alignment),
                        "Qwen3.5 executor workspace overflow");
  };

  add(checked_mul(max_context, sizeof(std::int32_t),
                  "token buffer byte size overflow"));
  add(sizeof(std::int32_t));
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
  add(max_attention_workspace(config, max_tokens, max_context));
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

class Qwen35Executor::Impl {
public:
  Impl(ExecutionContext &context, const model::Qwen35Config &config,
       const model::CudaWeightPlan &weights, std::size_t max_context)
      : context_(context), config_(config), weights_(weights),
        max_context_(max_context), dtype_(BRT_DTYPE_F32),
        weight_dtype_(dtype_from_weight(weights.token_embedding().type)),
        element_bytes_(dtype_size(dtype_)) {
    DeviceGuard guard{context_.device_id()};
    validate_config(config_, max_context_);
    require(weights_.layer_count() == config_.blocks.size(),
            "CUDA weight layer count does not match Qwen3.5 config");
    validate_weight_dtypes(weights_);
    allocate_buffers();
    create_plans();
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
      auto result = run_chunk(one, true, start_position);
      position_ = start_position + 1;
      return result;
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
    check_cuda(cudaStreamSynchronize(context_.stream()),
               "Qwen3.5 executor reset synchronization failed");
    position_ = 0;
    poisoned_ = false;
  }

  std::size_t position() const noexcept { return position_; }
  bool poisoned() const noexcept { return poisoned_; }

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

  void enable_trace(bool enabled) {
    trace_enabled_ = enabled;
    trace_.clear();
  }

  const std::vector<Qwen35TraceEntry> &trace() const noexcept { return trace_; }

private:
  friend class Qwen35Executor;

  struct FullState {
    float *kv_cache{};
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
    attention_workspace_ = allocate(
        arena, max_attention_workspace(config_, max_tokens, max_context_),
        alignof(float));
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
        const std::size_t kv_cache_bytes = checked_mul(
            checked_mul(max_context_, 2, "KV cache element count overflow"),
            checked_mul(full_kv_width, sizeof(float),
                        "KV cache byte size overflow"),
            "KV cache byte size overflow");
        full_states_.push_back(
            FullState{.kv_cache = static_cast<float *>(
                          allocate(arena, kv_cache_bytes, alignof(float))),
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
        plans.ffn_gate = CublasLtMatmulPlan::create(
            matmul_config(bucket, config_.intermediate_size,
                          config_.hidden_size, dtype_, weight_dtype_));
        plans.ffn_up = CublasLtMatmulPlan::create(
            matmul_config(bucket, config_.intermediate_size,
                          config_.hidden_size, dtype_, weight_dtype_));
        plans.ffn_down = CublasLtMatmulPlan::create(
            matmul_config(bucket, config_.hidden_size,
                          config_.intermediate_size, dtype_, weight_dtype_));
        if (block.kind == model::Qwen35BlockKind::full_attention) {
          require(weights.full_attention.has_value(),
                  "full-attention block is missing CUDA weights");
          const std::size_t q_width = config_.full_attention_head_count *
                                      config_.full_attention_head_dimension;
          const std::size_t kv_width = config_.full_attention_kv_head_count *
                                       config_.full_attention_head_dimension;
          plans.full = std::make_unique<FullPlans>();
          plans.full->query_gate = CublasLtMatmulPlan::create(matmul_config(
              bucket, q_width * 2, config_.hidden_size, dtype_, weight_dtype_));
          plans.full->key = CublasLtMatmulPlan::create(matmul_config(
              bucket, kv_width, config_.hidden_size, dtype_, weight_dtype_));
          plans.full->value = CublasLtMatmulPlan::create(matmul_config(
              bucket, kv_width, config_.hidden_size, dtype_, weight_dtype_));
          plans.full->output = CublasLtMatmulPlan::create(matmul_config(
              bucket, config_.hidden_size, q_width, dtype_, weight_dtype_));
        } else {
          require(weights.linear_attention.has_value(),
                  "linear-attention block is missing CUDA weights");
          const std::size_t qkv_width = linear_qkv_width(config_);
          const std::size_t linear_gate_width =
              config_.linear_value_head_count * config_.linear_head_dimension;
          plans.linear = std::make_unique<LinearPlans>();
          plans.linear->qkv = CublasLtMatmulPlan::create(matmul_config(
              bucket, qkv_width, config_.hidden_size, dtype_, weight_dtype_));
          plans.linear->beta = CublasLtMatmulPlan::create(
              matmul_config(bucket, config_.linear_value_head_count,
                            config_.hidden_size, dtype_, weight_dtype_));
          plans.linear->alpha = CublasLtMatmulPlan::create(
              matmul_config(bucket, config_.linear_value_head_count,
                            config_.hidden_size, dtype_, weight_dtype_));
          plans.linear->gate = CublasLtMatmulPlan::create(
              matmul_config(bucket, linear_gate_width, config_.hidden_size,
                            dtype_, weight_dtype_));
          plans.linear->output = CublasLtMatmulPlan::create(
              matmul_config(bucket, config_.hidden_size, linear_gate_width,
                            dtype_, weight_dtype_));
        }
      }
      entry.lm_head = CublasLtMatmulPlan::create(
          matmul_config(1, config_.vocabulary_size, config_.hidden_size, dtype_,
                        weight_dtype_));
    }
  }

  BucketPlans &bucket_for(std::size_t tokens) {
    for (auto &bucket : bucket_plans_) {
      if (bucket.tokens == tokens)
        return bucket;
    }
    throw Qwen35ExecutorError("unsupported Qwen3.5 executor token bucket");
  }

  void run_matmul(const CublasLtMatmulPlan &plan, const void *input,
                  const model::CudaTensorView &weight, void *output) {
    const std::size_t weight_element_bytes = dtype_size(weight_dtype_);
    require(plan.input_bytes() % weight_element_bytes == 0,
            "Qwen3.5 matmul input byte size is not dtype-aligned");
    kernels::qwen35_cast_f32(static_cast<const float *>(input), matmul_input_,
                             plan.input_bytes() / weight_element_bytes,
                             weight_dtype_, context_.stream());
    plan.run(context_.stream(), matmul_input_, plan.input_bytes(),
             weight.device_data, weight.bytes, output, plan.output_bytes(),
             matmul_workspace_, kMatmulWorkspaceBudget);
  }

  Qwen35ExecutorResult run_chunk(std::span<const std::int32_t> tokens,
                                 bool produce_logits,
                                 std::size_t chunk_position) {
    BucketPlans &bucket = bucket_for(tokens.size());
    check_cuda(cudaMemcpyAsync(device_tokens_ + chunk_position, tokens.data(),
                               tokens.size_bytes(), cudaMemcpyHostToDevice,
                               context_.stream()),
               "Qwen3.5 token upload failed");
    kernels::qwen35_embedding(
        device_tokens_ + chunk_position, weights_.token_embedding().device_data,
        hidden_a_,
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
                       *plans.full, chunk_position);
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
      run_matmul(*plans.ffn_gate, scratch, weights.common.ffn_gate,
                 intermediate_a_);
      run_matmul(*plans.ffn_up, scratch, weights.common.ffn_up,
                 intermediate_b_);
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
    kernels::qwen35_argmax_typed(logits_, device_result_,
                                 config_.vocabulary_size, dtype_,
                                 context_.stream());
    std::int32_t host_result = 0;
    check_cuda(cudaMemcpyAsync(&host_result, device_result_,
                               sizeof(host_result), cudaMemcpyDeviceToHost,
                               context_.stream()),
               "Qwen3.5 result download failed");
    check_cuda(cudaStreamSynchronize(context_.stream()),
               "Qwen3.5 executor synchronization failed");
    return Qwen35ExecutorResult{
        .token = host_result,
        .position =
            static_cast<std::uint32_t>(chunk_position + tokens.size() - 1),
    };
  }

  void run_full_layer(std::size_t tokens, const void *input, void *output,
                      const model::Qwen35CudaLayerWeights &weights,
                      const FullPlans &plans, std::size_t chunk_position) {
    const auto &full = *weights.full_attention;
    const std::size_t kv_width = config_.full_attention_kv_head_count *
                                 config_.full_attention_head_dimension;
    run_matmul(*plans.query_gate, input, full.query, full_query_gate_);
    run_matmul(*plans.key, input, full.key, full_key_);
    run_matmul(*plans.value, input, full.value, full_value_);
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
        dtype_from_weight(full.query_norm.type), context_.stream());
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
        context_.stream());
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
    kernels::qwen35_causal_attention(
        full_query_norm_, full_key_norm_, full_value_, linear_gate_,
        attention_out_, full_states_[state_index].kv_cache,
        static_cast<float *>(attention_workspace_),
        kernels::qwen35_attention_workspace_floats(attention_shape),
        attention_shape, dtype_, context_.stream());
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
    run_matmul(*plans.qkv, input, linear.qkv, linear_qkv_);
    record_trace("linear_attn_qkv_mixed-" + std::to_string(layer), linear_qkv_,
                 tokens * qkv_width);
    run_matmul(*plans.beta, input, linear.beta, linear_beta_);
    run_matmul(*plans.alpha, input, linear.alpha, linear_alpha_);
    run_matmul(*plans.gate, input, linear.gate, linear_gate_);
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
        dtype_, dtype_from_weight(linear.convolution.type), context_.stream());
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
  BrtDataType dtype_{};
  BrtDataType weight_dtype_{};
  std::size_t element_bytes_{};
  std::size_t position_{};
  bool poisoned_{};
  bool trace_enabled_{};

  std::int32_t *device_tokens_{};
  std::int32_t *device_result_{};
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
  void *delta_workspace_{};

  std::vector<FullState> full_states_;
  std::vector<LinearState> linear_states_;
  std::vector<std::size_t> full_state_by_layer_;
  std::vector<std::size_t> linear_state_by_layer_;
  std::vector<BucketPlans> bucket_plans_;
  std::vector<Qwen35TraceEntry> trace_;
};

std::size_t Qwen35Executor::workspace_bytes(const model::Qwen35Config &config,
                                            std::size_t max_context) {
  return workspace_estimate(config, max_context);
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
                               std::size_t max_context)
    : impl_(new Impl(context, config, weights, max_context)) {}

Qwen35Executor::~Qwen35Executor() noexcept { delete impl_; }

Qwen35ExecutorResult
Qwen35Executor::prefill(std::span<const std::int32_t> tokens) {
  return impl_->prefill(tokens);
}

Qwen35ExecutorResult Qwen35Executor::decode(std::int32_t token) {
  return impl_->decode(token);
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

} // namespace brt
