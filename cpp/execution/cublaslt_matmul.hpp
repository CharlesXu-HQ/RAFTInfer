#pragma once

#include <raftinfer/tensor.h>

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace raftinfer {

class CublasLtMatmulError : public std::runtime_error {
public:
  explicit CublasLtMatmulError(const std::string &message)
      : std::runtime_error(message) {}
};

enum class CublasLtMatrixOrder {
  RowMajor,
};

struct CublasLtMatmulShape {
  std::size_t m;
  std::size_t n;
  std::size_t k;
  bool transpose_input;
  bool transpose_weight;
};

struct CublasLtMatmulConfig {
  CublasLtMatmulShape shape;
  RaftInferDataType input_dtype;
  RaftInferDataType weight_dtype;
  RaftInferDataType output_dtype;
  CublasLtMatrixOrder input_order;
  CublasLtMatrixOrder weight_order;
  CublasLtMatrixOrder output_order;
  std::size_t workspace_budget_bytes;
};

struct CublasLtCandidate {
  cublasLtMatmulAlgo_t algorithm;
  int algorithm_id;
  std::size_t workspace_bytes;
};

namespace detail {

struct CublasLtRunBuffers {
  const void *input;
  std::size_t input_bytes;
  const void *weight;
  std::size_t weight_bytes;
  void *output;
  std::size_t output_bytes;
  void *workspace;
  std::size_t workspace_bytes;
};

struct CublasLtBufferRequirements {
  std::size_t input_bytes;
  std::size_t weight_bytes;
  std::size_t output_bytes;
  std::size_t workspace_bytes;
};

// Shared by plan creation and focused contract tests. These functions perform
// no CUDA work and allocate no memory.
void validate_cublaslt_shape(const CublasLtMatmulShape &shape);
void validate_cublaslt_run_buffers(
    const CublasLtRunBuffers &buffers,
    const CublasLtBufferRequirements &requirements);

} // namespace detail

std::vector<CublasLtCandidate>
enumerate_cublaslt_candidates(const CublasLtMatmulConfig &config,
                              std::size_t maximum = 16);

// Immutable cuBLASLt descriptor and algorithm plan. Creation performs device
// validation and the single heuristic query. `run` only validates caller-owned
// buffers and enqueues the saved algorithm on the supplied stream.
class CublasLtMatmulPlan {
public:
  static std::unique_ptr<CublasLtMatmulPlan>
  create(const CublasLtMatmulConfig &config);

  ~CublasLtMatmulPlan() noexcept;

  CublasLtMatmulPlan(const CublasLtMatmulPlan &) = delete;
  CublasLtMatmulPlan &operator=(const CublasLtMatmulPlan &) = delete;
  CublasLtMatmulPlan(CublasLtMatmulPlan &&) = delete;
  CublasLtMatmulPlan &operator=(CublasLtMatmulPlan &&) = delete;

  std::size_t input_bytes() const noexcept;
  std::size_t weight_bytes() const noexcept;
  std::size_t output_bytes() const noexcept;
  std::size_t workspace_bytes() const noexcept;
  int algorithm_id() const noexcept;

  void run(cudaStream_t stream, const void *input, std::size_t input_bytes,
           const void *weight, std::size_t weight_bytes, void *output,
           std::size_t output_bytes, void *workspace,
           std::size_t workspace_bytes) const;

  void select_fastest(cudaStream_t stream, const void *input,
                      const void *weight, void *output, void *workspace,
                      std::size_t workspace_bytes, std::uint32_t warmups = 2,
                      std::uint32_t measurements = 5);

private:
  class Impl;
  explicit CublasLtMatmulPlan(std::unique_ptr<Impl> impl) noexcept;
  friend std::vector<CublasLtCandidate>
  enumerate_cublaslt_candidates(const CublasLtMatmulConfig &config,
                                std::size_t maximum);

  std::unique_ptr<Impl> impl_;
};

} // namespace raftinfer
