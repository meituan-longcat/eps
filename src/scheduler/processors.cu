#include "src/scheduler/scheduler.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "src/scheduler/common.cuh"

namespace eps {

template <typename SchedulerTraits>
Scheduler<SchedulerTraits>::Processors::Processors(Params p, RunParams rp) {
    CommExpander::Params comm_expander_params{
      .num_tokens = rp.num_tokens,
      .topk = p.topk,
      .num_experts = p.num_experts,
      .num_stages = rp.num_stages,
      .ep_world_size = p.ep_world_size};
  comm_expander = CommExpander{comm_expander_params};

  AlltoallRunner::Params alltoall_runner_params{
      .my_rank = p.rank, .num_stages = rp.num_stages, .ep_world_size = p.ep_world_size,
      .num_ranks_per_node = p.num_ranks_per_node
      };
  alltoall_runner = AlltoallRunner{alltoall_runner_params};

  GemmExpander::Params gemm_expander_params{
      .local_expert_begin = p.rank * (p.num_experts / p.ep_world_size),
      .local_expert_end = (p.rank + 1) * (p.num_experts / p.ep_world_size),
      .num_experts = p.num_experts,
      .num_stages = rp.num_stages,
      .topk = p.topk,
      .ep_world_size = p.ep_world_size,
      .max_num_tokens_per_gpu = p.max_num_tokens_per_gpu,
      .hidden_size = p.hidden_size,
      .num_tokens_hint = std::max<int>(1, rp.num_global_tokens * p.topk / p.ep_world_size)
      };
  gemm_expander = GemmExpander{gemm_expander_params};

  LocalReducer::Params local_reducer_params{
      .local_expert_begin = p.rank * (p.num_experts / p.ep_world_size),
      .local_expert_end = (p.rank + 1) * (p.num_experts / p.ep_world_size),
      .num_experts = p.num_experts,
      .num_stages = rp.num_stages,
      .topk = p.topk,
      .ep_world_size = p.ep_world_size,
      .max_num_tokens_per_gpu = p.max_num_tokens_per_gpu,
      .hidden_size = p.hidden_size,
      .num_tokens_hint = std::max<int>(1, rp.num_global_tokens * p.topk / p.ep_world_size)};
  local_reducer = LocalReducer{local_reducer_params};

  GlobalReducer::Params global_reducer_params{
      .num_tokens = rp.num_tokens,
      .num_experts = p.num_experts,
      .num_stages = rp.num_stages,
      .topk = p.topk,
      .ep_world_size = p.ep_world_size,
      .hidden_size = p.hidden_size
  };
  global_reducer = GlobalReducer{global_reducer_params};

  typename ZeroReducerType::Params zero_reducer_params{
      .num_local_tokens = rp.num_tokens,
      .topk = p.topk,
      .hidden_size = p.hidden_size
  };
  zero_reducer = ZeroReducerType(zero_reducer_params);
}

template <typename SchedulerTraits>
size_t Scheduler<SchedulerTraits>::Processors::getWorkspaceSize() {
  size_t bytes = 0;

  ws.offsets.gemm_expander = bytes;
  size_t gemm_expander_bytes = gemm_expander.getWorkspaceSize();
  bytes += gemm_expander_bytes;

  ws.offsets.local_reducer = bytes;
  size_t local_reducer_bytes = local_reducer.getWorkspaceSize();
  bytes += local_reducer_bytes;

  ws.offsets.global_reducer = bytes;
  size_t global_reducer_bytes = global_reducer.getWorkspaceSize();
  bytes += global_reducer_bytes;

  ws.offsets.zero_reducer = bytes;
  size_t zero_reducer_bytes = zero_reducer.getWorkspaceSize();
  bytes += zero_reducer_bytes;

  ws.offsets.combine_alltoall_runner = bytes;
  size_t combine_alltoall_runner_bytes = alltoall_runner.getWorkspaceSize();
  bytes += combine_alltoall_runner_bytes;

  return bytes;
}

template <typename SchedulerTraits>
void Scheduler<SchedulerTraits>::Processors::set_ws(char* ws_ptr) {
    ws.ws_ = ws_ptr;
}

template <typename SchedulerTraits>
char *Scheduler<SchedulerTraits>::Processors::Workspace::gemm_expander()
{
    return ws_ + offsets.gemm_expander;
}

template <typename SchedulerTraits>
char *Scheduler<SchedulerTraits>::Processors::Workspace::local_reducer()
{
    return ws_ + offsets.local_reducer;
}

template <typename SchedulerTraits>
char *Scheduler<SchedulerTraits>::Processors::Workspace::global_reducer()
{
    return ws_ + offsets.global_reducer;
}

template <typename SchedulerTraits>
char *Scheduler<SchedulerTraits>::Processors::Workspace::combine_alltoall_runner()
{
    return ws_ + offsets.combine_alltoall_runner;
}

template <typename SchedulerTraits>
char *Scheduler<SchedulerTraits>::Processors::Workspace::zero_reducer()
{
    return ws_ + offsets.zero_reducer;
}

template
class Scheduler<SchedulerTraits<float, float, float>>::Processors;

template
class Scheduler<SchedulerTraits<half, half, half>>::Processors;
template
class Scheduler<SchedulerTraits<half, half, float>>::Processors;

#ifdef ENABLE_BF16
template
class Scheduler<SchedulerTraits<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>::Processors;
template
class Scheduler<SchedulerTraits<__nv_bfloat16, __nv_bfloat16, float>>::Processors;
#endif

template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, float, float>>::Processors;

template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, half, half>>::Processors;
template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, half, float>>::Processors;

template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16>>::Processors;
template
class Scheduler<SchedulerTraits<__nv_fp8_e4m3, __nv_bfloat16, float>>::Processors;

} // namespace eps