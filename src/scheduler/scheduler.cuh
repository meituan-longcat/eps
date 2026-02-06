#pragma once

#include "src/communication/alltoall_runner.cuh"
#include "src/scheduler/comm_expander.cuh"
#include "src/scheduler/gemm_expander.cuh"
#include "src/scheduler/local_reducer.cuh"
#include "src/scheduler/global_reducer.cuh"
#include "src/scheduler/zero_reducer.cuh"
#include "src/communication/msccl_comm.cuh"
#include "src/scheduler/gemm_executor_interface.cuh"
#include "src/scheduler/comm_quant.cuh"

namespace eps {

  template <typename _InputType,
            typename _OutputType,
            typename _ExpertScaleType,
            typename _GemmInputType = _InputType,
            typename _GemmOutputType = _OutputType,
            typename _CombineCommType = _OutputType>
  struct SchedulerTraits
  {
    using InputType = _InputType;
    using OutputType = _OutputType;
    using ExpertScaleType = _ExpertScaleType;
    using GemmInputType = _GemmInputType;
    using GemmOutputType = _GemmOutputType;
    using CombineCommType = _CombineCommType;
  };

template <typename SchedulerTraits>
class Scheduler {
 public:
  using Traits = SchedulerTraits;
  using CommBuffsType = CommBuffs<typename Traits::InputType, typename Traits::CombineCommType, typename Traits::ExpertScaleType>;

 public:
  struct Params {
    int topk;
    int num_experts;
    int64_t hidden_size;
    int input_scales_dim;
    int64_t max_num_tokens_per_gpu;
    MscclppCommunicator* comm;
    int max_num_stages;

    int rank;
    int ep_world_size;
    int num_ranks_per_node;

    Params(int _topk, int _num_experts, int64_t _hidden_size, int _input_scales_dim, int64_t _max_num_tokens_per_gpu, MscclppCommunicator* _comm, int _max_num_stages=1);
  };

  struct RunParams {
    int num_tokens;
    int num_global_tokens;
    int num_stages;
    bool input_quanted;
  };

  struct Processors {
    struct Workspace {
      struct WorkspaceOffsets {
        size_t gemm_expander;
        size_t local_reducer;
        size_t global_reducer;
        size_t combine_alltoall_runner;
        size_t zero_reducer;
      };

      char* gemm_expander();
      char* local_reducer();
      char* global_reducer();
      char* combine_alltoall_runner();
      char* zero_reducer();

      char* ws_{};
      WorkspaceOffsets offsets;
    };

    Processors(Params p, RunParams rp);

    size_t getWorkspaceSize();

    void set_ws(char* ws_ptr);

    CommExpander comm_expander;
    GemmExpander gemm_expander;
    LocalReducer local_reducer;
    GlobalReducer global_reducer;
    AlltoallRunner alltoall_runner;
    using ZeroReducerType = ZeroReducer<typename SchedulerTraits::InputType, typename SchedulerTraits::OutputType, typename SchedulerTraits::ExpertScaleType>;
    ZeroReducerType zero_reducer;

    Workspace ws;
  };

  struct Context {
    struct WorkspaceOffsets {
      size_t processors;
    } workspace_offsets;

    Context(Params p, RunParams _rp);

    size_t getWorkspaceSize();

    void set_ws(char* ws_ptr);

    char* processors_ws();

    RunParams rp;
    Processors processors;
    char* ws{};

    ExpandForCommPlan expand_for_comm_plan;
    CommPlan dispatch_comm_plan;
    ExpandForGemmPlan expand_for_gemm_plan;
    LocalReducePlan local_reduce_plan;
    CommPlan  combine_comm_plan;
    GlobalReducePlan global_reduce_plan;
    ZeroReducePlan<typename SchedulerTraits::InputType, typename SchedulerTraits::ExpertScaleType> zero_reduce_plan;

    Params p_;
    int num_will_recv_unique_tokens_;
  };

  Scheduler(Params p);

  ~Scheduler();

  Context* getContext(RunParams rp);

  int rendezvous(const int *expert_indices, Context &context, cudaStream_t stream);

  void plan(const int* expert_indices,
    const typename SchedulerTraits::ExpertScaleType* expert_scales,
    int* exclusive_sum /*[local_expert_end - local_expert_begin + 1]*/,
    Context& context,
    cudaStream_t stream);

  void prologue(const typename SchedulerTraits::ExpertScaleType *expert_scales,
                const typename SchedulerTraits::InputType *hidden_states,
                const float *input_scales,
                float* recv_input_scales,
                Context &context,
                cudaStream_t stream);

  void dispatch_comm(Context& context, int stage_idx, cudaStream_t stream);

  void gemm_expand(typename SchedulerTraits::GemmInputType *gemm_input,
                   Context &context,
                   int stage_idx,
                   cudaStream_t stream);

  void local_reduce(typename SchedulerTraits::GemmOutputType *gemm_output,
                    Context &context,
                    int stage_idx,
                    cudaStream_t stream);

  void combine_comm(Context& context, int stage_idx, cudaStream_t stream);

  void epilogue(typename SchedulerTraits::OutputType *output,
                Context &context,
                cudaStream_t stream);

  void barrier() const {
    if (p_.comm == nullptr) {
      throw std::runtime_error("barrier cannot be called on nullptr MscclppCommunicator!");
    }
    p_.comm->barrier();
  }

 public:
  std::shared_ptr<CommBuffsType> comm_buffs_;

 private:
  Params p_;

  // Host-side MoE info
  volatile int* num_will_recv_unique_tokens_ = nullptr;
  int* num_will_recv_unique_tokens_mapped_ = nullptr;

  char* comm_expander_ws_ = nullptr;
  size_t comm_expander_ws_bytes_ = 0;

  char* dispatch_alltoall_runner_ws_ = nullptr;
  size_t dispatch_alltoall_runner_ws_bytes_ = 0;
};

} // namespace eps