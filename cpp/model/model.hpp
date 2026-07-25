#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "gguf_types.hpp"
#include "qwen35_config.hpp"

namespace brt::model {

class ModelIoError : public std::runtime_error {
public:
  explicit ModelIoError(const std::string &message)
      : std::runtime_error(message) {}
};

class Model {
public:
  explicit Model(const std::string &gguf_path);
  ~Model();

  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  Model(Model &&) = delete;
  Model &operator=(Model &&) = delete;

  std::span<const std::uint8_t> tokenizer_spec() const noexcept;
  const Qwen35Config &qwen35_config() const noexcept;
  std::span<const std::uint8_t>
  tensor_payload(const gguf::TensorInfo &tensor) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace brt::model
