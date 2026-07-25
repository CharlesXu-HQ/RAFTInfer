#pragma once

#include "../model/model.hpp"
#include "qwen35_state.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace brt {

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

class Session {
public:
  Session(std::shared_ptr<const model::Model> model,
          std::uint32_t max_context_tokens);
  ~Session() noexcept;

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  const model::Model &model() const noexcept;
  const Qwen35HostState &host_state() const noexcept;
  SessionTokenResult prefill(std::span<const std::int32_t> tokens);
  SessionTokenResult decode(std::int32_t token);
  void reset();

private:
  std::shared_ptr<const model::Model> model_;
  Qwen35HostState state_;
#if BRT_ENABLE_CUDA
  std::unique_ptr<DeviceExecutionOwner> execution_owner_;
  std::unique_ptr<Qwen35Executor> executor_;
#endif
};

} // namespace brt
