#include "../model/model.hpp"
#include "../reference/qwen35_executor.hpp"

#include "assert_enabled.hpp"
#include "qwen35_nonzero_fixture.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

class TemporaryFixture {
public:
  TemporaryFixture()
      : path_(std::filesystem::temp_directory_path() /
              ("brt_qwen35_cpu_reference_" +
               std::to_string(static_cast<long long>(getpid())) + ".gguf")) {
    const auto bytes = brt::test::make_qwen35_nonzero_bf16_gguf_fixture();
    std::ofstream output{path_, std::ios::binary};
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.close();
    assert(output);
  }

  ~TemporaryFixture() noexcept {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::filesystem::path &path() const noexcept { return path_; }

private:
  std::filesystem::path path_;
};

void assert_valid_execution(
    const brt::reference::Qwen35ReferenceExecution &execution) {
  assert(execution.logits.size() == 16);
  assert(std::all_of(execution.logits.begin(), execution.logits.end(),
                     [](float value) { return std::isfinite(value); }));
  assert(execution.token >= 0);
  assert(execution.token < 16);

  auto sorted = execution.logits;
  std::sort(sorted.begin(), sorted.end(), std::greater<float>{});
  assert(sorted[0] - sorted[1] > 5.0e-2F);
}

} // namespace

int main() {
  TemporaryFixture fixture;
  brt::model::Model model{fixture.path().string()};

  const std::vector<std::int32_t> prompt{1, 2, 3, 4};
  const auto prefill = brt::reference::qwen35_execute_model(model, prompt);
  assert_valid_execution(prefill);

  const std::vector<std::int32_t> prompt_and_decode{1, 2, 3, 4, 5};
  const auto decode =
      brt::reference::qwen35_execute_model(model, prompt_and_decode);
  assert_valid_execution(decode);
  assert(prefill.logits != decode.logits);
}
