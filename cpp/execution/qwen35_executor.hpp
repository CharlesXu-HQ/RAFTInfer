#pragma once

#include "execution_context.hpp"

#include "../model/cuda_weights.hpp"
#include "../model/qwen35_config.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

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

class Qwen35Executor {
public:
  static constexpr std::size_t workspace_alignment = 256;

  static std::size_t workspace_bytes(const model::Qwen35Config &config,
                                     std::size_t max_context);
  static void validate_request(const model::Qwen35Config &config,
                               std::size_t past_tokens,
                               std::span<const std::int32_t> tokens);
  static void
  validate_weight_dtypes_for_tests(const model::CudaWeightPlan &weights);

  Qwen35Executor(ExecutionContext &context, const model::Qwen35Config &config,
                 const model::CudaWeightPlan &weights, std::size_t max_context);
  ~Qwen35Executor() noexcept;

  Qwen35Executor(const Qwen35Executor &) = delete;
  Qwen35Executor &operator=(const Qwen35Executor &) = delete;
  Qwen35Executor(Qwen35Executor &&) = delete;
  Qwen35Executor &operator=(Qwen35Executor &&) = delete;

  Qwen35ExecutorResult prefill(std::span<const std::int32_t> tokens);
  Qwen35ExecutorResult decode(std::int32_t token);
  void copy_last_logits(std::span<float> output) const;
  void reset();

  std::size_t position() const noexcept;
  bool poisoned() const noexcept;

private:
  class Impl;
  Impl *impl_{};
};

} // namespace brt
