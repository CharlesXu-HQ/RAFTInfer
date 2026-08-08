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

class ScopedDecodeAttentionEnvironment {
public:
  explicit ScopedDecodeAttentionEnvironment(const char *value) {
    if (const char *current = std::getenv("RAFTINFER_QWEN35_DECODE_ATTENTION")) {
      original_ = current;
      had_original_ = true;
    }
    if (value == nullptr) {
      assert(unsetenv("RAFTINFER_QWEN35_DECODE_ATTENTION") == 0);
    } else {
      assert(setenv("RAFTINFER_QWEN35_DECODE_ATTENTION", value, 1) == 0);
    }
  }

  ~ScopedDecodeAttentionEnvironment() {
    if (had_original_) {
      assert(setenv("RAFTINFER_QWEN35_DECODE_ATTENTION", original_.c_str(), 1) ==
             0);
    } else {
      assert(unsetenv("RAFTINFER_QWEN35_DECODE_ATTENTION") == 0);
    }
  }

private:
  std::string original_;
  bool had_original_{};
};

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

void run_decode_attention_environment_tests() {
  using Mode = raftinfer::Qwen35DecodeAttentionMode;
  const struct {
    const char *value;
    Mode expected;
  } accepted[] = {
      {"auto", Mode::auto_select},
      {"single-block", Mode::single_block},
      {"split-k-256", Mode::split_k_256},
      {"split-k-512", Mode::split_k_512},
  };
  for (const auto &[value, expected] : accepted) {
    ScopedDecodeAttentionEnvironment environment{value};
    auto policy = raftinfer::Qwen35ExecutionPolicy{};
    policy.decode_attention = Mode::single_block;
    assert(raftinfer::qwen35_execution_policy_from_environment(policy)
               .decode_attention == expected);
  }
  {
    ScopedDecodeAttentionEnvironment environment{nullptr};
    auto policy = raftinfer::Qwen35ExecutionPolicy{};
    policy.decode_attention = Mode::split_k_512;
    assert(raftinfer::qwen35_execution_policy_from_environment(policy)
               .decode_attention == Mode::split_k_512);
  }
  for (const char *value : {"", "split-k", "SPLIT-K-256"}) {
    ScopedDecodeAttentionEnvironment environment{value};
    assert_throws<std::invalid_argument>([] {
      (void)raftinfer::qwen35_execution_policy_from_environment({});
    });
  }
}

} // namespace

int main() {
  run_decode_attention_environment_tests();
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
