#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "src/common/torch_utils.h"
#include "src/executor/activations.cuh"

#include "silu_ops.h"

namespace {

void run(
    const at::Tensor &gate_up,
    at::Tensor &result,
    const at::Tensor &exclusive_sum,
    int64_t num_tokens_hint
) {
  EPS_CHECK_TENSOR(2, gate_up);
  EPS_CHECK_TENSOR(2, result);
  EPS_CHECK_TENSOR(1, exclusive_sum);
  TORCH_CHECK(exclusive_sum.scalar_type() == at::kInt, "exclusive_sum must be of type kInt32");

  int num_experts = exclusive_sum.size(0) - 1;
  int inter_size_x2 = gate_up.size(1);

  eps::silu_activate<__nv_bfloat16>(
    (__nv_bfloat16*)gate_up.data_ptr(),
    (__nv_bfloat16*)result.data_ptr(),
    exclusive_sum.data_ptr<int32_t>(),
    num_experts,
    inter_size_x2,
    num_tokens_hint,
    at::cuda::getCurrentCUDAStream()
  );
}

} // namespace

void executor::register_silu_ops(torch::Library &m) {
  m.def("silu_run", &run);
}
