#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "src/common/torch_utils.h"
#include "src/scheduler/scheduler.cuh"

#include "scheduler_ops.h"

namespace {

using fptr_t = int64_t;

using Scheduler = eps::Scheduler<eps::SchedulerTraits<__nv_bfloat16, __nv_bfloat16, float>>;

fptr_t create(
    int64_t topk,
    int64_t num_experts,
    int64_t hidden_size,
    int64_t input_scales_dim,
    int64_t max_num_tokens_per_gpu,
    fptr_t comm,
    int64_t max_num_stages
) {
  auto *ptr = new Scheduler(
    Scheduler::Params{
        (int)topk,
        (int)num_experts,
        hidden_size,
        (int)input_scales_dim,
        max_num_tokens_per_gpu,
        (eps::MscclppCommunicator *)comm,
        (int)max_num_stages
    });
  return (fptr_t)ptr;
}

void destroy(fptr_t ptr) { delete (Scheduler *)ptr; }

fptr_t get_context(
  fptr_t ptr,
  int64_t num_tokens,
  int64_t num_global_tokens,
  int64_t num_stages,
  bool input_quanted
) {
  auto *scheduler = (Scheduler *)ptr;

  return (fptr_t)(scheduler->getContext(
    Scheduler::RunParams{
      .num_tokens = (int)num_tokens,
      .num_global_tokens = (int)num_global_tokens,
      .num_stages = (int)num_stages,
      .input_quanted = input_quanted
    }
  ));
}

void destroy_context(fptr_t ptr) {  delete (Scheduler::Context *)ptr; }

int64_t get_workspace_size_context(fptr_t ptr) {
  return ((Scheduler::Context *)ptr)->getWorkspaceSize();
}

void set_ws_context(fptr_t ptr, at::Tensor &ws) {
  ((Scheduler::Context *)ptr)->set_ws((char*)ws.data_ptr());
}

int64_t rendezvous(
    fptr_t ptr,
    const at::Tensor &expert_indices,
    fptr_t context)
{
  EPS_CHECK_TENSOR(2, expert_indices);

  auto *scheduler = (Scheduler *)ptr;

  return scheduler->rendezvous(
      expert_indices.data_ptr<int32_t>(),
      *(Scheduler::Context *)context,
      at::cuda::getCurrentCUDAStream());
}

void plan(
    fptr_t ptr,
    const at::Tensor &expert_indices,
    const at::Tensor &expert_scales,
    const at::Tensor &exclusive_sum,
    fptr_t context)
{
  EPS_CHECK_TENSOR(2, expert_indices);
  EPS_CHECK_TENSOR(1, exclusive_sum);

  auto *scheduler = (Scheduler *)ptr;

  scheduler->plan(
      expert_indices.data_ptr<int32_t>(),
      (typename Scheduler::Traits::ExpertScaleType*)expert_scales.data_ptr(),
      exclusive_sum.data_ptr<int32_t>(),
      *(Scheduler::Context *)context,
      at::cuda::getCurrentCUDAStream());
}

void prologue(
    fptr_t ptr,
    const at::Tensor &expert_scales,
    const at::Tensor &hidden_states,
    const std::optional<at::Tensor> &input_scales,
    const std::optional<at::Tensor> &recv_input_scales,
    fptr_t context)
{
  EPS_CHECK_TENSOR(2, expert_scales);
  EPS_CHECK_TENSOR(2, hidden_states);

  auto *scheduler = (Scheduler *)ptr;

  scheduler->prologue(
      (typename Scheduler::Traits::ExpertScaleType*)expert_scales.data_ptr(),
      (typename Scheduler::Traits::InputType*)hidden_states.data_ptr(),
      (const float*)(input_scales.has_value() ? input_scales->data_ptr() : nullptr),
      (float *)(recv_input_scales.has_value() ? recv_input_scales->data_ptr() : nullptr),
      *(Scheduler::Context *)context,
      at::cuda::getCurrentCUDAStream());
}

void dispatch(
    fptr_t ptr,
    fptr_t context,
    int64_t stage_idx
) {
  auto *scheduler = (Scheduler *)ptr;

  scheduler->dispatch_comm(
    *(Scheduler::Context *)context,
    stage_idx,
    at::cuda::getCurrentCUDAStream()
  );
}

void gemm_expand(
    fptr_t ptr,
    at::Tensor &gemm_input,
    fptr_t context,
    int64_t stage_idx
) {
  EPS_CHECK_TENSOR(2, gemm_input);

  auto *scheduler = (Scheduler *)ptr;

  scheduler->gemm_expand(
    (typename Scheduler::Traits::GemmInputType*)gemm_input.data_ptr(),
    *(Scheduler::Context *)context,
    stage_idx,
    at::cuda::getCurrentCUDAStream()
  );
}

void local_reduce(
    fptr_t ptr,
    at::Tensor &gemm_output,
    fptr_t context,
    int64_t stage_idx
) {
  EPS_CHECK_TENSOR(2, gemm_output);

  auto *scheduler = (Scheduler *)ptr;

  scheduler->local_reduce(
    (typename Scheduler::Traits::GemmOutputType*)gemm_output.data_ptr(),
    *(Scheduler::Context *)context,
    stage_idx,
    at::cuda::getCurrentCUDAStream()
  );
}

void combine(
    fptr_t ptr,
    fptr_t context,
    int64_t stage_idx
) {
  auto *scheduler = (Scheduler *)ptr;

  scheduler->combine_comm(
    *(Scheduler::Context *)context,
    stage_idx,
    at::cuda::getCurrentCUDAStream()
  );
}

void epilogue(
    fptr_t ptr,
    at::Tensor &output,
    fptr_t context)
{
  EPS_CHECK_TENSOR(2, output);

  auto *scheduler = (Scheduler *)ptr;

  scheduler->epilogue(
      (typename Scheduler::Traits::OutputType*)output.data_ptr(),
      *(Scheduler::Context *)context,
      at::cuda::getCurrentCUDAStream());
}

} // namespace

void eps::register_scheduler_ops(torch::Library &m) {
  m.def("eps_create", &create);
  m.def("eps_destroy", &destroy);
  m.def("eps_get_context", &get_context);
  m.def("eps_destroy_context", &destroy_context);
  m.def("eps_get_workspace_size_context", &get_workspace_size_context);
  m.def("eps_set_ws_context", &set_ws_context);
  m.def("eps_rendezvous", &rendezvous);
  m.def("eps_plan", &plan);
  m.def("eps_prologue", &prologue);
  m.def("eps_dispatch", &dispatch);
  m.def("eps_gemm_expand", &gemm_expand);
  m.def("eps_local_reduce", &local_reduce);
  m.def("eps_combine", &combine);
  m.def("eps_epilogue", &epilogue);
}
