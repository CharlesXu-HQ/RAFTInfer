#pragma once

#include "../model/gguf_reader.hpp"
#include "../reference/bf16.hpp"

#include "qwen35_gguf_fixture.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace brt::test {

inline std::vector<std::uint8_t>
make_qwen35_nonzero_bf16_gguf_fixture(Qwen35GgufFixtureOptions options = {}) {
  constexpr std::uint32_t kBf16TensorType = 30;
  auto bytes = make_qwen35_gguf_fixture(kBf16TensorType, false, options);
  const auto catalog = gguf::read_catalog(bytes);

  for (std::size_t tensor_index = 0; tensor_index < catalog.tensors.size();
       ++tensor_index) {
    const auto &tensor = catalog.tensors[tensor_index];
    assert(tensor.byte_size % sizeof(std::uint16_t) == 0);
    const std::size_t elements =
        static_cast<std::size_t>(tensor.byte_size / sizeof(std::uint16_t));
    const std::size_t payload_offset =
        static_cast<std::size_t>(catalog.tensor_data_offset + tensor.offset);
    assert(payload_offset + static_cast<std::size_t>(tensor.byte_size) <=
           bytes.size());

    for (std::size_t element = 0; element < elements; ++element) {
      const float phase =
          static_cast<float>((tensor_index + 1) * 17 + (element + 1) * 13);
      float value =
          0.1875F * std::sin(phase * 0.173F) +
          static_cast<float>(static_cast<int>(element % 7) - 3) / 64.0F;
      if (tensor.name == "output.weight") {
        const std::size_t row = element / options.hidden_size;
        const std::size_t col = element % options.hidden_size;
        value = 0.5F * std::sin(static_cast<float>((row + 1) * (col + 3)) *
                                0.319F) +
                static_cast<float>(static_cast<int>(row) - 7) / 128.0F;
      }
      const std::uint16_t encoded = reference::float_to_bf16(value).bits;
      const std::size_t offset =
          payload_offset + element * sizeof(std::uint16_t);
      bytes[offset] = static_cast<std::uint8_t>(encoded & 0xffU);
      bytes[offset + 1] = static_cast<std::uint8_t>((encoded >> 8U) & 0xffU);
    }
  }
  return bytes;
}

} // namespace brt::test
