#pragma once

#include "gguf_types.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace brt::gguf {

class ParseError : public std::runtime_error {
public:
  explicit ParseError(const std::string &message)
      : std::runtime_error(message) {}
};

Catalog read_catalog(std::span<const std::uint8_t> bytes,
                     const ReaderLimits &limits = {});

} // namespace brt::gguf
