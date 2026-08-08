#pragma once

#include "../model/model.hpp"
#include "qwen35_execution_policy.hpp"
#include "qwen35_state.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace raftinfer {

class DeviceExecutionOwner;
class Qwen35Executor;

class SessionUnavailableError : public std::runtime_error {
public:
  explicit SessionUnavailableError(const std::string &message)
      : std::runtime_error(message) {}
};

class SessionCudaError : public std::runtime_error {
public:
  explicit SessionCudaError(const std::string &message)
      : std::runtime_error(message) {}
};

struct SessionTokenResult {
  std::int32_t token_id{};
  std::uint32_t position{};
};

struct SessionDiagnostics {
  Qwen35AttentionImplementation attention{
      Qwen35AttentionImplementation::online_tiled};
  Qwen35KvCacheDType kv_cache{Qwen35KvCacheDType::f32};
  Qwen35KvCacheLayout kv_cache_layout{Qwen35KvCacheLayout::token_major};
  bool decode_graph_enabled{};
  bool decode_graph_captured{};
  bool decode_graph_replayed{};
  std::size_t attention_workspace_bytes{};
  Qwen35DecodeAttentionMode decode_attention{
      Qwen35DecodeAttentionMode::single_block};
  std::size_t decode_attention_partition_tokens{};
  std::size_t decode_attention_threshold_tokens{};
  std::size_t decode_attention_context_bucket_tokens{};
  bool decode_attention_split_k_graph_captured{};
};

class Session {
public:
  Session(std::shared_ptr<const model::Model> model,
          std::uint32_t max_context_tokens,
          Qwen35ExecutionPolicy policy = {});
  ~Session() noexcept;

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  const model::Model &model() const noexcept;
  const Qwen35HostState &host_state() const noexcept;
  SessionTokenResult prefill(std::span<const std::int32_t> tokens);
  SessionTokenResult decode(std::int32_t token);
  SessionTokenResult decode_greedy(std::int32_t first_token,
                                   std::span<std::int32_t> output_tokens);
  SessionDiagnostics diagnostics() const;
  void reset();

private:
  std::shared_ptr<const model::Model> model_;
  Qwen35ExecutionPolicy policy_;
  Qwen35HostState state_;
#if RAFTINFER_ENABLE_CUDA
  std::unique_ptr<DeviceExecutionOwner> execution_owner_;
  std::unique_ptr<Qwen35Executor> executor_;
#endif
};

} // namespace raftinfer
