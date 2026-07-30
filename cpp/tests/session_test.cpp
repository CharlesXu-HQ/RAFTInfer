#include "../execution/session.hpp"
#include "../model/model.hpp"

#include "assert_enabled.hpp"
#include "qwen35_gguf_fixture.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

class TemporaryFile {
public:
  explicit TemporaryFile(const std::vector<std::uint8_t> &bytes) {
    auto pattern =
        (std::filesystem::temp_directory_path() / "raftinfer-session-XXXXXX")
            .string();
    const int descriptor = mkstemp(pattern.data());
    assert(descriptor >= 0);
    close(descriptor);
    path_ = std::move(pattern);
    std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    assert(stream.good());
  }

  ~TemporaryFile() { std::filesystem::remove(path_); }

  const std::string &path() const noexcept { return path_; }

private:
  std::string path_;
};

template <class Exception, class Callable>
void assert_throws(Callable &&callable) {
  bool threw = false;
  try {
    callable();
  } catch (const Exception &) {
    threw = true;
  }
  assert(threw);
}

} // namespace

int main() {
  const TemporaryFile fixture(raftinfer::test::make_qwen35_gguf_fixture());
  auto model = std::make_shared<raftinfer::model::Model>(fixture.path());
  raftinfer::Session session(model, 8);

  assert(!session.host_state().has_tensor_storage());
  assert(session.host_state().position() == 0);
  assert(session.host_state().full_kv_length(3) == 0);

  model.reset();
  assert(session.model().qwen35_config().blocks.size() == 4);

  const std::int32_t valid_tokens[] = {1, 2, 3};
  assert_throws<raftinfer::SessionUnavailableError>([&] {
    (void)session.prefill(std::span<const std::int32_t>{valid_tokens});
  });
  assert(session.host_state().position() == 0);

  assert_throws<std::invalid_argument>(
      [&] { (void)session.decode(-1); });
  assert_throws<std::invalid_argument>(
      [&] { (void)session.decode(16); });
  assert(session.host_state().position() == 0);

  assert_throws<raftinfer::SessionUnavailableError>(
      [&] { (void)session.decode(3); });
  assert(session.host_state().position() == 0);

  session.reset();
  assert(session.host_state().position() == 0);
  assert(session.host_state().full_kv_length(3) == 0);
}
