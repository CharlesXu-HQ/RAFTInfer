#include "session.hpp"

#include <stdexcept>
#include <utility>

namespace brt {

Session::Session(std::shared_ptr<const model::Model> model,
                 std::uint32_t max_context_tokens)
    : model_(std::move(model)),
      state_(Qwen35StateLayout::create(model_ ? model_->qwen35_config()
                                              : throw std::invalid_argument(
                                                    "model is required"),
                                      max_context_tokens)) {}

const model::Model &Session::model() const noexcept { return *model_; }

const Qwen35HostState &Session::host_state() const noexcept { return state_; }

void Session::reset() noexcept { state_.reset(); }

} // namespace brt
