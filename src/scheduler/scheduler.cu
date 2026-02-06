#include "src/scheduler/scheduler.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "src/common/debug.cuh"

#include <cuda_profiler_api.h>

namespace eps {

template <typename SchedulerTraits>
Scheduler<SchedulerTraits>::Params::Params(int _topk, int _num_experts, int64_t _hidden_size, int _input_scales_dim, int64_t _max_num_tokens_per_gpu, MscclppCommunicator* _comm, int _max_num_stages):
  topk{_topk},
  num_experts{_num_experts},
  hidden_size{_hidden_size},
  input_scales_dim{_input_scales_dim},
  max_num_tokens_per_gpu{_max_num_tokens_per_gpu},
  comm{_comm},
  max_num_stages{_max_num_stages},
  rank{_comm->getRank()},
  ep_world_size{_comm->getWorldSize()},
  num_ranks_per_node{_comm->getNumRanksPerNode()}
{

}

template <typename SchedulerTraits>
Scheduler<SchedulerTraits>::Scheduler(Params p) : p_{p} {
  typename CommBuffsType::Params comm_buffs_params{
    .topk = p_.topk,
    .max_num_tokens_per_gpu = p_.max_num_tokens_per_gpu,
    .hidden_size = p_.hidden_size,
    .input_scales_dim = p_.input_scales_dim,
    .ep_world_size = p_.ep_world_size,
    .max_num_stages = p_.max_num_stages,
    .comm = p_.comm
  };

  comm_buffs_ = std::make_shared<CommBuffsType>(comm_buffs_params);

  cudaMallocHost(&num_will_recv_unique_tokens_, sizeof(int), cudaHostAllocMapped);
  cudaHostGetDevicePointer(&num_will_recv_unique_tokens_mapped_, const_cast<int*>(num_will_recv_unique_tokens_), 0);
  *num_will_recv_unique_tokens_ = -1;
}

template <typename SchedulerTraits>
Scheduler<SchedulerTraits>::~Scheduler() {
}

template <typename SchedulerTraits>
Scheduler<SchedulerTraits>::Context* Scheduler<SchedulerTraits>::getContext(RunParams rp) {
  if ((p_.num_experts / p_.ep_world_size) % rp.num_stages == 0)
  {
    return new Context{p_, rp};
  }
  else
  {
    throw std::runtime_error("Invalid num_stages: " + std::to_string(rp.num_stages) + " for local_num_experts: " + std::to_string(p_.num_experts / p_.ep_world_size));
  }
}

template <typename SchedulerTraits>
void Scheduler<SchedulerTraits>::prologue(
  const typename SchedulerTraits::ExpertScaleType* expert_scales,
  const typename Traits::InputType* hidden_states,
  const float* input_scales,
  float* recv_input_scales,
  Context& context,
  cudaStream_t stream) {
    sync_check_cuda_error_eps();
    auto& processors = context.processors;

    context.zero_reduce_plan.setRawInput(hidden_states);

    processors.comm_expander.run(
      expert_scales,
      comm_buffs_->expert_scales.send_buf,
      p_.topk,
      context.expand_for_comm_plan,
      stream
    );
    sync_check_cuda_error_eps();

    using Dtype = typename decltype(comm_buffs_->expert_scales)::Dtype;
    processors.alltoall_runner.template run<Dtype>(
      comm_buffs_->expert_scales,
      context.dispatch_comm_plan,
      p_.topk,
      -1,
      stream
    );
    sync_check_cuda_error_eps();

    processors.comm_expander.run(
      hidden_states,
      comm_buffs_->input_hidden.send_buf,
      p_.hidden_size,
      context.expand_for_comm_plan,
      stream
    );
    sync_check_cuda_error_eps();

    if (context.rp.input_quanted) {
      processors.comm_expander.run(
        input_scales,
        comm_buffs_->input_scales.send_buf,
        p_.input_scales_dim,
        context.expand_for_comm_plan,
        stream
      );
      sync_check_cuda_error_eps();

      processors.alltoall_runner.template run<float>(
        comm_buffs_->input_scales,
        context.dispatch_comm_plan,
        p_.input_scales_dim,
        -1,
        stream
      );
      sync_check_cuda_error_eps();

      processors.gemm_expander.run(
        comm_buffs_->input_scales.recv_buf,
        recv_input_scales,
        p_.input_scales_dim,
        context.expand_for_gemm_plan,
        -1,
        stream
      );
    }
}

template <typename SchedulerTraits>
void Scheduler<SchedulerTraits>::dispatch_comm(Context& context, int stage_idx, cudaStream_t stream){
    auto& processors = context.processors;

    using Dtype = typename decltype(comm_buffs_->input_hidden)::Dtype;
    processors.alltoall_runner.template run<Dtype>(
      comm_buffs_->input_hidden,
      context.dispatch_comm_plan,
      p_.hidden_size,
      stage_idx,
      stream
    );
    sync_check_cuda_error_eps();
}

template <typename SchedulerTraits>
void Scheduler<SchedulerTraits>::gemm_expand(
    typename SchedulerTraits::GemmInputType *gemm_input,
    Context &context,
    int stage_idx,
    cudaStream_t stream)
{
  auto &processors = context.processors;

  processors.gemm_expander.run(
      comm_buffs_->input_hidden.recv_buf,
      gemm_input,
      p_.hidden_size,
      context.expand_for_gemm_plan,
      stage_idx,
      stream);
  sync_check_cuda_error_eps();

}

template <typename SchedulerTraits>
void Scheduler<SchedulerTraits>::local_reduce(
    typename SchedulerTraits::GemmOutputType *gemm_output,
    Context &context,
    int stage_idx,
    cudaStream_t stream)
{
  auto &processors = context.processors;

  processors.local_reducer.run(
      gemm_output,
      comm_buffs_->output_hidden.send_buf,
      comm_buffs_->expert_scales.recv_buf,
      context.local_reduce_plan,
      stage_idx,
      stream);
  sync_check_cuda_error_eps();
}

template <typename SchedulerTraits>
void Scheduler<SchedulerTraits>::combine_comm(Context& context, int stage_idx, cudaStream_t stream){
    auto& processors = context.processors;

    using Dtype = typename decltype(comm_buffs_->output_hidden)::Dtype;
    processors.alltoall_runner.template run<Dtype>(
      comm_buffs_->output_hidden,
      context.combine_comm_plan,
      p_.hidden_size,
      stage_idx,
      stream
    );
    sync_check_cuda_error_eps();
}

template <typename SchedulerTraits>
void Scheduler<SchedulerTraits>::epilogue(
    typename SchedulerTraits::OutputType *output,
    Context &context,
    cudaStream_t stream)
{
  auto &processors = context.processors;

  processors.global_reducer.run(
      comm_buffs_->output_hidden.recv_buf,
      output,
      context.global_reduce_plan,
      (typename SchedulerTraits::OutputType *)nullptr,
      stream);
  sync_check_cuda_error_eps();
  
  processors.zero_reducer.run(output, context.zero_reduce_plan, stream);
  sync_check_cuda_error_eps();

  const int num_nodes = p_.ep_world_size / p_.num_ranks_per_node;
  if (num_nodes > 1)
  {
    processors.alltoall_runner.template flush<int>(comm_buffs_->dispatch_comm_plan, stream);
  }
}

template <typename SchedulerTraits>
int Scheduler<SchedulerTraits>::rendezvous(
  const int *expert_indices,
  Context& context,
  cudaStream_t stream) {
    auto& processors = context.processors;

    if (processors.comm_expander.getWorkspaceSize() > comm_expander_ws_bytes_) {
      if (comm_expander_ws_ != nullptr) {
        cudaFreeAsync(comm_expander_ws_, stream);
      }

      comm_expander_ws_bytes_ = processors.comm_expander.getWorkspaceSize();
      cudaMallocAsync(&comm_expander_ws_, comm_expander_ws_bytes_, stream);
    }

    if (processors.alltoall_runner.getWorkspaceSize() > dispatch_alltoall_runner_ws_bytes_) {
      if (dispatch_alltoall_runner_ws_ != nullptr) {
        cudaFreeAsync(dispatch_alltoall_runner_ws_, stream);
      }

      dispatch_alltoall_runner_ws_bytes_ = processors.alltoall_runner.getWorkspaceSize();
      cudaMallocAsync(&dispatch_alltoall_runner_ws_, dispatch_alltoall_runner_ws_bytes_, stream);
    }

    auto expand_for_comm_plan = processors.comm_expander.plan(
      expert_indices,
      comm_buffs_->dispatch_comm_plan.send_buf,
      comm_expander_ws_,
      stream
    );
    context.expand_for_comm_plan = expand_for_comm_plan;
    sync_check_cuda_error_eps();

    *num_will_recv_unique_tokens_ = -1;
    auto dispatch_comm_plan = processors.alltoall_runner.plan(
      comm_buffs_->dispatch_comm_plan,
      dispatch_alltoall_runner_ws_,
      num_will_recv_unique_tokens_mapped_,
      stream
    );
    sync_check_cuda_error_eps();
    context.dispatch_comm_plan = dispatch_comm_plan;

    while (true) {
      if (static_cast<int>(*num_will_recv_unique_tokens_) >= 0) {
        break;
      }
    }

    context.num_will_recv_unique_tokens_ = *num_will_recv_unique_tokens_;

    return context.num_will_recv_unique_tokens_;
}

template <typename SchedulerTraits>
void Scheduler<SchedulerTraits>::plan(
    const int *expert_indices,
    const typename SchedulerTraits::ExpertScaleType *expert_scales,
    int *exclusive_sum,
    Context &context,
    cudaStream_t stream)
{
  auto &processors = context.processors;
  processors.comm_expander.run(
      expert_indices,
      comm_buffs_->expert_indices.send_buf,
      p_.topk,
      context.expand_for_comm_plan,
      stream);
  sync_check_cuda_error_eps();

  processors.alltoall_runner.template run<int>(
      comm_buffs_->expert_indices,
      context.dispatch_comm_plan,
      p_.topk,
      -1,
      stream);
  sync_check_cuda_error_eps();

  auto expand_for_gemm_plan = processors.gemm_expander.plan(
      comm_buffs_->expert_indices.recv_buf,
      context.dispatch_comm_plan.recv_plan,
      exclusive_sum,
      processors.ws.gemm_expander(),
      stream,
      p_.rank);
  sync_check_cuda_error_eps();

  auto local_reduce_plan = processors.local_reducer.plan(
      comm_buffs_->expert_indices.recv_buf,
      comm_buffs_->combine_comm_plan.send_buf,
      context.dispatch_comm_plan.recv_plan,
      expand_for_gemm_plan,
      processors.ws.local_reducer(),
      stream,
      context.rp.num_global_tokens);
  sync_check_cuda_error_eps();

  auto global_reduce_plan = processors.global_reducer.plan(
      expert_indices,
      context.expand_for_comm_plan,
      processors.ws.global_reducer(),
      stream);
  sync_check_cuda_error_eps();

  auto combine_comm_plan = processors.alltoall_runner.plan(
      comm_buffs_->combine_comm_plan,
      processors.ws.combine_alltoall_runner(),
      nullptr,
      stream
      );
  sync_check_cuda_error_eps();

  auto zero_reducer_plan =
      processors.zero_reducer.plan(expert_indices, expert_scales, processors.ws.zero_reducer(), stream);
  sync_check_cuda_error_eps();
  context.zero_reduce_plan = zero_reducer_plan;

  context.expand_for_gemm_plan = expand_for_gemm_plan;
  context.local_reduce_plan = local_reduce_plan;
  context.combine_comm_plan = combine_comm_plan;
  context.global_reduce_plan = global_reduce_plan;
}

template
class Scheduler<SchedulerTraits<float, float, float>>;

template
class Scheduler<SchedulerTraits<half, half, half>>;
template
class Scheduler<SchedulerTraits<half, half, float>>;

#ifdef ENABLE_BF16
template
class Scheduler<SchedulerTraits<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>;
template
class Scheduler<SchedulerTraits<__nv_bfloat16, __nv_bfloat16, float>>;
template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, __nv_bfloat16, float>>;
template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16>>;
#endif

template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, float, float>>;

template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, half, half>>;
template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, half, float>>;

}
