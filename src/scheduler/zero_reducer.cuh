#pragma once

#include <iostream>

#include "src/common/common.cuh"
#include "src/scheduler/comm_expander.cuh"

namespace eps {

template <typename InputType, typename ExpertScaleType>
struct ZeroReducePlan {
  ExpertScaleType* zero_reduce_scales;
  const InputType* raw_input;
  void setRawInput(const InputType* _raw_input) { raw_input = _raw_input; }
};

template <typename InputType, typename ExpertScaleType>
struct ZeroReducePlanKernelParams {
  const int* expert_indices;
  const ExpertScaleType* expert_scales;
  int num_local_tokens;
  int topk;
  ZeroReducePlan<InputType, ExpertScaleType> zero_reduce_plan;
};

template <typename InputType, typename OutputType, typename ExpertScaleType>
struct ZeroReduceKernelParams {
  ExpertScaleType* zero_reduce_scales;
  int num_local_tokens;
  int64_t hidden_size;
  const InputType* raw_input;
  OutputType* output;
};

template <typename InputType, typename ExpertScaleType>
void getZeroReducePlan(ZeroReducePlanKernelParams<InputType, ExpertScaleType> p, cudaStream_t stream);

template <typename InputType, typename OutputType, typename ExpertScaleType>
void zeroReduce(ZeroReduceKernelParams<InputType, OutputType, ExpertScaleType> p, cudaStream_t stream);

template <typename InputType, typename OutputType, typename ExpertScaleType>
class ZeroReducer {
 public:
  struct Params {
    int num_local_tokens;
    int topk;
    int64_t hidden_size;
  };

  ZeroReducer() = default;

  ZeroReducer(Params p) : p_{p} {}

  size_t getWorkspaceSize() {
    size_t zero_reduce_scales = p_.num_local_tokens * sizeof(ExpertScaleType);
    zero_reduce_scales = ALIGN(zero_reduce_scales);
    return zero_reduce_scales;
  }

  ZeroReducePlan<InputType, ExpertScaleType> plan(const int* expert_indices,
                                                  const ExpertScaleType* expert_scales,
                                                  void* ws,
                                                  cudaStream_t stream) {
    ZeroReducePlan<InputType, ExpertScaleType> zero_reduce_plan{.zero_reduce_scales = (ExpertScaleType*)ws,
                                                                .raw_input = nullptr};
    ZeroReducePlanKernelParams<InputType, ExpertScaleType> p{.expert_indices = expert_indices,
                                                             .expert_scales = expert_scales,
                                                             .num_local_tokens = p_.num_local_tokens,
                                                             .topk = p_.topk,
                                                             .zero_reduce_plan = zero_reduce_plan};
    getZeroReducePlan(p, stream);
    return zero_reduce_plan;
  }

  void run(OutputType* output, ZeroReducePlan<InputType, ExpertScaleType> zero_reduce_plan, cudaStream_t stream) {
    ZeroReduceKernelParams<InputType, OutputType, ExpertScaleType> p{
        .zero_reduce_scales = zero_reduce_plan.zero_reduce_scales,
        .num_local_tokens = p_.num_local_tokens,
        .hidden_size = p_.hidden_size,
        .raw_input = zero_reduce_plan.raw_input,
        .output = output};

    zeroReduce(p, stream);
  }

 private:
  Params p_{};
};

}  // namespace eps