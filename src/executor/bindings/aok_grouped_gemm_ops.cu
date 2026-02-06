#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "src/common/torch_utils.h"
#include "src/executor/aok_grouped_gemm.cuh"

#include "aok_grouped_gemm_ops.h"

namespace {

using fptr_t = int64_t;

using AokGroupedGemm = eps::AokGroupedGemm<__nv_bfloat16, __nv_bfloat16, int>;

fptr_t create(
    int64_t num_local_experts,
    int64_t sm_count
) {
  auto *ptr = new AokGroupedGemm(
    AokGroupedGemm::Params{
    .num_local_experts = (int)num_local_experts,
    .sm_count = (int)sm_count
  });

  return (fptr_t)ptr;
}

void destroy(fptr_t ptr) { delete (AokGroupedGemm *)ptr; }

int64_t get_workspace_size(fptr_t ptr, int64_t cols, int64_t max_num_tokens) {
  auto *gemm = (AokGroupedGemm *)ptr;
  return gemm->getWorkspaceSize(cols, max_num_tokens, gemm->p_.num_local_experts);
}

void run(
    fptr_t ptr,
    const at::Tensor &A,
    const at::Tensor &B,
    at::Tensor &C,
    const at::Tensor &exclusive_sum,
    int64_t num_tokens_hint,
    int64_t max_num_tokens,
    at::Tensor &ws
) {
  EPS_CHECK_TENSOR(2, A);
  EPS_CHECK_TENSOR(3, B);
  EPS_CHECK_TENSOR(2, C);
  EPS_CHECK_TENSOR(1, exclusive_sum);
  TORCH_CHECK(exclusive_sum.scalar_type() == at::kInt, "exclusive_sum must be of type kInt32");

  auto gemm_k = A.size(1);
  auto gemm_n = B.size(1);
  int local_num_experts  = B.size(0);

  typename GemmInterface::RunParams p{
      .A = A.data_ptr(),
      .A_scales = nullptr,
      .B = B.data_ptr(),
      .weight_scales = nullptr,
      .biases = nullptr,
      .C = C.data_ptr(),
      .total_rows_before_expert = exclusive_sum.data_ptr<int32_t>(),
      .total_rows = num_tokens_hint, // used as num_tokens_hint
      .gemm_n = gemm_n,
      .gemm_k = gemm_k,
      .num_experts_per_stage = local_num_experts,
      .expert_begin = 0,
      .ws = (char*)ws.data_ptr(),
      .max_num_tokens = max_num_tokens
  };

  auto *gemm = (AokGroupedGemm *)ptr;
  gemm->run(p, at::cuda::getCurrentCUDAStream());
}

} // namespace

void executor::register_aok_gemm_ops(torch::Library &m) {
  m.def("aok_create", &create);
  m.def("aok_get_workspace_size", &get_workspace_size);
  m.def("aok_run", &run);
  m.def("aok_destroy", &destroy);
}
