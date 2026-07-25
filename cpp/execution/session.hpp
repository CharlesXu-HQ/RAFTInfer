#pragma once

#include "../model/model.hpp"
#include "qwen35_state.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace brt {

class Session {
public:
  Session(std::shared_ptr<const model::Model> model,
          std::uint32_t max_context_tokens);

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  const model::Model &model() const noexcept;
  const Qwen35HostState &host_state() const noexcept;
  void reset() noexcept;

private:
  std::shared_ptr<const model::Model> model_;
  Qwen35HostState state_;
};

} // namespace brt
