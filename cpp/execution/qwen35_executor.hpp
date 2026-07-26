#pragma once

#include "execution_context.hpp"
#include "qwen35_execution_policy.hpp"

#include "../model/cuda_weights.hpp"
#include "../model/qwen35_config.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace brt {

class Qwen35ExecutorError : public std::runtime_error {
public:
  explicit Qwen35ExecutorError(const std::string &message)
      : std::runtime_error(message) {}
};

struct Qwen35ExecutorResult {
  std::int32_t token{};
  std::uint32_t position{};
};

struct Qwen35ExecutionDiagnostics {
  Qwen35AttentionImplementation attention{};
  Qwen35KvCacheDType kv_cache_dtype{};
  Qwen35KvCacheLayout kv_cache_layout{};
  bool decode_graph_captured{};
  bool decode_graph_replayed{};
  std::size_t attention_workspace_bytes{};
};

struct Qwen35TraceEntry {
  std::string name;
  double sum{};
  std::array<float, 3> first{};
  std::array<float, 3> last{};
};

class Qwen35Executor {
public:
  static constexpr std::size_t workspace_alignment = 256;

  static std::size_t workspace_bytes(const model::Qwen35Config &config,
                                     std::size_t max_context,
                                     Qwen35ExecutionPolicy policy = {});
  static void validate_request(const model::Qwen35Config &config,
                               std::size_t past_tokens,
                               std::span<const std::int32_t> tokens);
  static void
  validate_weight_dtypes_for_tests(const model::CudaWeightPlan &weights);

  Qwen35Executor(ExecutionContext &context, const model::Qwen35Config &config,
                 const model::CudaWeightPlan &weights, std::size_t max_context,
                 Qwen35ExecutionPolicy policy = {});
  ~Qwen35Executor() noexcept;

  Qwen35Executor(const Qwen35Executor &) = delete;
  Qwen35Executor &operator=(const Qwen35Executor &) = delete;
  Qwen35Executor(Qwen35Executor &&) = delete;
  Qwen35Executor &operator=(Qwen35Executor &&) = delete;

  Qwen35ExecutorResult prefill(std::span<const std::int32_t> tokens);
  Qwen35ExecutorResult decode(std::int32_t token);
  void copy_last_logits(std::span<float> output) const;
  void enable_trace(bool enabled);
  const std::vector<Qwen35TraceEntry> &trace() const noexcept;
  void reset();

  std::size_t position() const noexcept;
  bool poisoned() const noexcept;
  Qwen35ExecutionDiagnostics diagnostics() const noexcept;

private:
  class Impl;
  Impl *impl_{};
};

} // namespace brt
