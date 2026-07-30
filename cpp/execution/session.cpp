#include "session.hpp"

#if RAFTINFER_ENABLE_CUDA
#include "qwen35_executor.hpp"

#include "../foundation/device_context.hpp"
#include "../model/cuda_weights.hpp"
#endif

#include <stdexcept>
#include <utility>

namespace raftinfer {
namespace {

Qwen35StateLayout
create_session_layout(const std::shared_ptr<const model::Model> &model,
                      std::uint32_t max_context_tokens) {
  if (!model) {
    throw std::invalid_argument("model is required");
  }
  if (max_context_tokens > model->qwen35_config().context_length) {
    throw std::invalid_argument(
        "max_context_tokens exceeds model context length");
  }
  return Qwen35StateLayout::create(model->qwen35_config(),
                                   max_context_tokens);
}

void validate_request(const model::Qwen35Config &config,
                      const Qwen35HostState &state,
                      std::span<const std::int32_t> tokens) {
  if (tokens.empty()) {
    throw std::invalid_argument("Qwen3.5 token span must not be empty");
  }
  const auto position = state.position();
  const auto max_context = state.layout().max_context_tokens;
  if (tokens.size() > max_context - position) {
    throw std::invalid_argument("Qwen3.5 request exceeds session context");
  }
  for (const auto token : tokens) {
    if (token < 0) {
      throw std::invalid_argument("Qwen3.5 token id is negative");
    }
    if (static_cast<std::size_t>(token) >= config.vocabulary_size) {
      throw std::invalid_argument("Qwen3.5 token id is out of range");
    }
  }
}

} // namespace

Session::Session(std::shared_ptr<const model::Model> model,
                 std::uint32_t max_context_tokens,
                 Qwen35ExecutionPolicy policy)
    : model_(std::move(model)),
      policy_(policy),
      state_(create_session_layout(model_, max_context_tokens),
             Qwen35HostStorage::LogicalOnly) {
#if RAFTINFER_ENABLE_CUDA
  if (!model_->cuda_ready()) {
    return;
  }
  try {
    const auto device = model_->device_context();
    const auto *weights = model_->cuda_weights();
    if (!device || weights == nullptr) {
      throw std::logic_error("CUDA model attachment is incomplete");
    }
    const auto workspace_bytes = Qwen35Executor::workspace_bytes(
        model_->qwen35_config(), max_context_tokens, policy_);
    execution_owner_ = device->create_execution_owner(
        workspace_bytes);
    auto context = execution_owner_->execution_context();
    executor_ = std::make_unique<Qwen35Executor>(
        context, model_->qwen35_config(), *weights, max_context_tokens,
        policy_);
  } catch (const std::bad_alloc &) {
    throw;
  } catch (const std::exception &error) {
    throw SessionCudaError(error.what());
  }
#endif
}

Session::~Session() noexcept = default;

const model::Model &Session::model() const noexcept { return *model_; }

const Qwen35HostState &Session::host_state() const noexcept { return state_; }

SessionTokenResult
Session::prefill(std::span<const std::int32_t> tokens) {
  validate_request(model_->qwen35_config(), state_, tokens);
#if RAFTINFER_ENABLE_CUDA
  if (executor_) {
    try {
      const auto result = executor_->prefill(tokens);
      state_.commit_tokens(static_cast<std::uint32_t>(tokens.size()));
      return SessionTokenResult{.token_id = result.token,
                                .position = result.position};
    } catch (const std::bad_alloc &) {
      throw;
    } catch (const std::exception &error) {
      throw SessionCudaError(error.what());
    }
  }
#endif
  throw SessionUnavailableError(
      "Qwen3.5 prefill backend requires a CUDA-loaded model");
}

SessionTokenResult Session::decode(std::int32_t token) {
  const std::span<const std::int32_t> tokens{&token, 1};
  validate_request(model_->qwen35_config(), state_, tokens);
#if RAFTINFER_ENABLE_CUDA
  if (executor_) {
    try {
      const auto result = executor_->decode(token);
      state_.commit_tokens(1);
      return SessionTokenResult{.token_id = result.token,
                                .position = result.position};
    } catch (const std::bad_alloc &) {
      throw;
    } catch (const std::exception &error) {
      throw SessionCudaError(error.what());
    }
  }
#endif
  throw SessionUnavailableError(
      "Qwen3.5 decode backend requires a CUDA-loaded model");
}

SessionTokenResult
Session::decode_greedy(std::int32_t first_token,
                       std::span<std::int32_t> output_tokens) {
  if (output_tokens.empty()) {
    throw std::invalid_argument("Qwen3.5 greedy decode output span is empty");
  }
  if (output_tokens.size() > state_.layout().max_context_tokens -
                                 state_.position()) {
    throw std::invalid_argument("Qwen3.5 request exceeds session context");
  }
  const std::span<const std::int32_t> tokens{&first_token, 1};
  validate_request(model_->qwen35_config(), state_, tokens);
#if RAFTINFER_ENABLE_CUDA
  if (executor_) {
    try {
      const auto result = executor_->decode_greedy(first_token, output_tokens);
      state_.commit_tokens(static_cast<std::uint32_t>(output_tokens.size()));
      return SessionTokenResult{.token_id = result.token,
                                .position = result.position};
    } catch (const std::bad_alloc &) {
      throw;
    } catch (const std::exception &error) {
      throw SessionCudaError(error.what());
    }
  }
#endif
  throw SessionUnavailableError(
      "Qwen3.5 decode backend requires a CUDA-loaded model");
}

SessionDiagnostics Session::diagnostics() const {
#if RAFTINFER_ENABLE_CUDA
  if (executor_) {
    const auto diagnostics = executor_->diagnostics();
    return SessionDiagnostics{
        .attention = diagnostics.attention,
        .kv_cache = diagnostics.kv_cache_dtype,
        .kv_cache_layout = diagnostics.kv_cache_layout,
        .decode_graph_enabled =
            policy_.decode_graph &&
            diagnostics.attention == Qwen35AttentionImplementation::online_tiled,
        .decode_graph_captured = diagnostics.decode_graph_captured,
        .decode_graph_replayed = diagnostics.decode_graph_replayed,
        .attention_workspace_bytes = diagnostics.attention_workspace_bytes,
    };
  }
#endif
  throw SessionUnavailableError(
      "Qwen3.5 execution diagnostics require a CUDA-loaded model");
}

void Session::reset() {
#if RAFTINFER_ENABLE_CUDA
  if (executor_) {
    try {
      executor_->reset();
    } catch (const std::exception &error) {
      throw SessionCudaError(error.what());
    }
  }
#endif
  state_.reset();
}

} // namespace raftinfer
