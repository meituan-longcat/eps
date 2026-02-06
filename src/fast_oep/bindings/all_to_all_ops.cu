#include "all_to_all_ops.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "src/common/torch_utils.h"
#include "all_to_all/fast_oep.cuh"

using fptr_t = int64_t;
using FastOEP = fast_oep::FastOEP<__nv_bfloat16>;

namespace {

fptr_t create(
    int64_t embed_dim,
    int64_t size_per_rank,
    int64_t max_num_global_tokens,
    fptr_t comm
) {
  auto *ptr = new FastOEP(
      FastOEP::Params{
          .embed_dim = embed_dim,
          .size_per_rank = size_per_rank,
          .max_num_global_tokens = max_num_global_tokens,
          .comm = (MscclppCommunicator *)comm});
  return (fptr_t)ptr;
}

void destroy(fptr_t ptr) { delete (FastOEP *)ptr; }

void dispatch(
    fptr_t ptr,
    at::Tensor &ids
) {
  EPS_CHECK_TENSOR(2, ids);
  TORCH_CHECK(ids.scalar_type() == at::kInt, "indices must be of type kInt32");

  auto *all_to_all = (FastOEP *)ptr;
  all_to_all->dispatch(
    ids.data_ptr<int32_t>(),
    ids.size(0) * ids.size(1),
    at::cuda::getCurrentCUDAStream()
  );
}

void combine(
    fptr_t ptr,
    at::Tensor &out,
    int64_t num_global_tokens,
    int64_t n_grams,
    bool do_permute,
    at::Tensor &embed_table
) {
  EPS_CHECK_TENSOR(3, out);
  TORCH_CHECK(
      out.scalar_type() == at::kBFloat16,
      "out must be of type BFloat16"
  );

  auto *all_to_all = (FastOEP *)ptr;
  all_to_all->combine(
    (__nv_bfloat16 *)out.data_ptr(),
    out.size(0) * out.size(1),
    num_global_tokens,
    n_grams,
    do_permute,
    (const __nv_bfloat16 *)embed_table.data_ptr(),
    at::cuda::getCurrentCUDAStream()
  );
}

} // namespace

void fast_oep::register_all_to_all_ops(torch::Library &m) {
  m.def("all_to_all_create", &create);
  m.def("all_to_all_destroy", &destroy);
  m.def("all_to_all_dispatch", &dispatch);
  m.def("all_to_all_combine", &combine);
}
