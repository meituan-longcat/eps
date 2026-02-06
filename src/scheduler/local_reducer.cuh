#pragma once

#include "src/scheduler/gemm_expander.cuh"

namespace eps {

struct LocalReducePlan {
  int* mapping; /*[num_tokens, topk]*/
  int* scale_mapping; /*[num_tokens]*/
  int* inclusive_sum; /*[ep_world_size, num_stages]*/
};

struct LocalReducePlanKernelParams {
  const int* expert_indices; /*[num_tokens, topk]*/
  int* tailing_experts; /*[num_tokens]*/
  int* sorted_tailing_experts; /*[num_tokens]*/
  const int* inclusive_sum; /*[ep_world_size, num_stages]*/
  int* mapping; /*[num_tokens]*/
  LocalReducePlan reduce_plan;
  ExpandForGemmPlan expand_plan;
  int local_expert_begin;
  int local_expert_end;
  int num_experts;
  int num_stages;
  int topk;
  int ep_world_size;
  int64_t max_num_tokens_per_gpu;
};

template <typename T, typename U>
struct LocalReduceKernelParams {
  LocalReducePlan reduce_plan;
  int num_stages;
  int topk;
  int ep_world_size;
  int64_t hidden_size;
  const T* src;
  T* dst;
  const U* expert_scales; /*[num_tokens, topk]*/
  int64_t num_tokens_hint;
};

void getLocalReducePlan(LocalReducePlanKernelParams p, cudaStream_t stream);

void getLocalReducePlan_ver2(LocalReducePlanKernelParams p, cudaStream_t stream, int num_global_tokens);

template <typename T, typename U>
void localReduce(LocalReduceKernelParams<T, U> p, int stage_idx, cudaStream_t stream);

class LocalReducer {
 public:
  struct Params {
    int local_expert_begin;
    int local_expert_end;
    int num_experts;
    int num_stages;
    int topk;
    int ep_world_size;
    int64_t max_num_tokens_per_gpu;
    int64_t hidden_size;
    int64_t num_tokens_hint;
  };

  LocalReducer() = default;

  LocalReducer(Params p) : p_{p} {}

  size_t getWorkspaceSize() {
    int mapping = p_.max_num_tokens_per_gpu * p_.ep_world_size  * p_.topk;
    int scale_mapping = p_.max_num_tokens_per_gpu * p_.ep_world_size;
    int tailing_experts = p_.max_num_tokens_per_gpu * p_.ep_world_size;
    int sorted_tailing_experts = p_.max_num_tokens_per_gpu * p_.ep_world_size;
    int _mapping = p_.max_num_tokens_per_gpu * p_.ep_world_size;
    size_t bytes = sizeof(int) * (mapping + scale_mapping + tailing_experts + sorted_tailing_experts + _mapping);
    return bytes;
  }

  LocalReducePlan plan(
      const int* expert_indices /*[num_tokens, topk]*/,
      int* inclusive_sum /*[ep_world_size, num_stages]*/,
      int* recv_plan /*[ep_world_size, num_stages]*/,
      ExpandForGemmPlan expand_plan,
      void* ws,
      cudaStream_t stream,
      int num_global_tokens) {
    int* mapping = (int*)ws;
    int* scale_mapping = mapping + p_.max_num_tokens_per_gpu * p_.ep_world_size * p_.topk;
    int* tailing_experts = scale_mapping + p_.max_num_tokens_per_gpu * p_.ep_world_size;
    int* sorted_tailing_experts = tailing_experts + p_.max_num_tokens_per_gpu * p_.ep_world_size;
    int* _mapping = sorted_tailing_experts + p_.max_num_tokens_per_gpu * p_.ep_world_size;

    LocalReducePlan reduce_plan{.mapping = mapping, .scale_mapping = scale_mapping, .inclusive_sum = inclusive_sum};

    LocalReducePlanKernelParams p{
        .expert_indices = expert_indices,
        .tailing_experts = tailing_experts,
        .sorted_tailing_experts = sorted_tailing_experts,
        .inclusive_sum = recv_plan,
        .mapping = _mapping,
        .reduce_plan = reduce_plan,
        .expand_plan = expand_plan,
        .local_expert_begin = p_.local_expert_begin,
        .local_expert_end = p_.local_expert_end,
        .num_experts = p_.num_experts,
        .num_stages = p_.num_stages,
        .topk = p_.topk,
        .ep_world_size = p_.ep_world_size,
        .max_num_tokens_per_gpu = p_.max_num_tokens_per_gpu};

    getLocalReducePlan_ver2(p, stream, num_global_tokens);
    // getLocalReducePlan(p, stream);

    return reduce_plan;
  }

  template <typename T, typename U>
  void
  run(const T* src, T* dst, const U* expert_scales, LocalReducePlan reduce_plan, int stage_idx, cudaStream_t stream) {
    LocalReduceKernelParams<T, U> p{
        .reduce_plan = reduce_plan,
        .num_stages = p_.num_stages,
        .topk = p_.topk,
        .ep_world_size = p_.ep_world_size,
        .hidden_size = p_.hidden_size,
        .src = src,
        .dst = dst,
        .expert_scales = expert_scales,
        .num_tokens_hint = p_.num_tokens_hint};

    localReduce(p, stage_idx, stream);
  }

 private:
  Params p_{};
};

} // namespace eps