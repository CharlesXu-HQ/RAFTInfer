#pragma once

#include <bit>
#include <cstdint>

namespace brt::reference {

struct bf16_t {
  std::uint16_t bits{};
};

inline float bf16_to_float(bf16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value.bits) << 16U);
}

inline bf16_t float_to_bf16(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  const std::uint32_t rounded = bits + 0x7FFFU + lsb;
  return bf16_t{static_cast<std::uint16_t>(rounded >> 16U)};
}

}  // namespace brt::reference
