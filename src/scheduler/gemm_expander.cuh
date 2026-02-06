#pragma once

#include <iostream>
#include "src/common/debug.cuh"

namespace eps {

struct ExpandForGemmPlan {
  int* mapping; /*[num_tokens * topk]*/
  int* exclusive_sum; /*[local_expert_end - local_expert_begin + 1]*/
};

struct ExpandForGemmPlanKernelParams {
  int* expert_indices; /*[num_tokens, topk]*/
  int* sorted_expert_indices; /*[num_tokens, topk]*/
  const int* inclusive_sum; /*[ep_world_size, num_stages]*/
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
struct ExpandForGemmKernelParams {
  ExpandForGemmPlan expand_plan;
  int local_expert_begin;
  int local_expert_end;
  int num_stages;
  int topk;
  int64_t cols;
  const T* src;
  U* dst;
  int64_t num_tokens_hint;
};

void getExpandForGemmPlan(ExpandForGemmPlanKernelParams p, cudaStream_t stream);

void getExpandForGemmPlan_ver2(ExpandForGemmPlanKernelParams p, cudaStream_t stream);

template <typename T, typename U>
void expandForGemm(ExpandForGemmKernelParams<T, U> p, int stage_idx, cudaStream_t stream);

class GemmExpander {
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

  GemmExpander() = default;

  GemmExpander(Params p) : p_{p} {}

  size_t getWorkspaceSize() {
    size_t mapping = p_.max_num_tokens_per_gpu * p_.ep_world_size * p_.topk;
    size_t sorted_expert_indices = p_.max_num_tokens_per_gpu * p_.ep_world_size * p_.topk;
    size_t bytes = (mapping + sorted_expert_indices) * sizeof(int);
    return bytes;
  }

  ExpandForGemmPlan plan(
      int* expert_indices /*[num_tokens, topk]*/,
      int* recv_plan /*[ep_world_size, num_stages]*/,
      int* exclusive_sum,
      void* ws,
      cudaStream_t stream,
      int global_rank) {
    int* mapping = (int*)ws;
    int* sorted_expert_indices = mapping + p_.max_num_tokens_per_gpu * p_.ep_world_size * p_.topk;

    ExpandForGemmPlan expand_plan{
        .mapping = mapping,
        .exclusive_sum = exclusive_sum,
    };

    ExpandForGemmPlanKernelParams p{
        .expert_indices = expert_indices,
        .sorted_expert_indices = sorted_expert_indices,
        .inclusive_sum = recv_plan,
        .expand_plan = expand_plan,
        .local_expert_begin = p_.local_expert_begin,
        .local_expert_end = p_.local_expert_end,
        .num_experts = p_.num_experts,
        .num_stages = p_.num_stages,
        .topk = p_.topk,
        .ep_world_size = p_.ep_world_size,
        .max_num_tokens_per_gpu = p_.max_num_tokens_per_gpu,
    };

    getExpandForGemmPlan_ver2(p, stream);
    return expand_plan;
  }

  template <typename T, typename U>
  void run(const T* src, U* dst, int64_t cols, ExpandForGemmPlan expand_plan, int stage_idx, cudaStream_t stream) {
    ExpandForGemmKernelParams<T, U> p{
        .expand_plan = expand_plan,
        .local_expert_begin = p_.local_expert_begin,
        .local_expert_end = p_.local_expert_end,
        .num_stages = p_.num_stages,
        .topk = p_.topk,
        .cols = cols,
        .src = src,
        .dst = dst,
        .num_tokens_hint = p_.num_tokens_hint};

    expandForGemm(p, stage_idx, stream);
  }

 private:
  Params p_{};
};

} // namespace eps