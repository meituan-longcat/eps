#include "all_to_all_ops.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "src/common/torch_utils.h"
#include "all_to_all/fast_ep.cuh"

using fptr_t = int64_t;
using FastEP = fast_ep::FastEP<__nv_bfloat16>;

namespace {

fptr_t create(
    int64_t topk,
    int64_t num_experts,
    int64_t hidden_size,
    int64_t max_num_global_tokens,
    fptr_t comm
) {
  auto *ptr = new FastEP(
      FastEP::Params{
          .topk = topk,
          .num_experts = num_experts,
          .hidden_size = hidden_size,
          .max_num_global_tokens = max_num_global_tokens,
          .comm = (MscclppCommunicator *)comm});
  return (fptr_t)ptr;
}

void destroy(fptr_t ptr) { delete (FastEP *)ptr; }

void dispatch(
    fptr_t ptr,
    at::Tensor &outExclusiveSum,
    at::Tensor &outExpertX,
    const at::Tensor &dpX,
    const at::Tensor &indices,
    int64_t num_global_tokens
) {
  EPS_CHECK_TENSOR(2, outExpertX);
  EPS_CHECK_TENSOR(2, dpX);
  EPS_CHECK_TENSOR(2, indices);
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be of type kInt32");

  auto *all_to_all = (FastEP *)ptr;
  all_to_all->dispatch(
    outExclusiveSum.data_ptr<int32_t>(),
    (__nv_bfloat16 *)outExpertX.data_ptr(),
    (__nv_bfloat16 *)dpX.data_ptr(),
    indices.data_ptr<int32_t>(),
    indices.size(0),
    num_global_tokens,
    at::cuda::getCurrentCUDAStream()
  );
}

void combine(
    fptr_t ptr,
    at::Tensor &outTokens,
    const at::Tensor &weights,
    const at::Tensor &expertY,
    int64_t num_global_tokens
) {
  EPS_CHECK_TENSOR(2, outTokens);
  TORCH_CHECK(
      outTokens.scalar_type() == at::kBFloat16,
      "outTokens must be of type BFloat16"
  );

  EPS_CHECK_TENSOR(2, weights);
  TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be of type Float32");
  EPS_CHECK_TENSOR(2, expertY);

  auto *all_to_all = (FastEP *)ptr;
  all_to_all->combine(
    (__nv_bfloat16 *)outTokens.data_ptr(),
    (const float *)weights.data_ptr(),
    (const __nv_bfloat16 *)expertY.data_ptr(),
    num_global_tokens,
    at::cuda::getCurrentCUDAStream()
  );
}

} // namespace

void fast_ep::register_all_to_all_ops(torch::Library &m) {
  m.def("all_to_all_create", &create);
  m.def("all_to_all_destroy", &destroy);
  m.def("all_to_all_dispatch", &dispatch);
  m.def("all_to_all_combine", &combine);
}
