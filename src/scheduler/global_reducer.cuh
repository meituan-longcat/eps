#pragma once

#include "src/scheduler/comm_expander.cuh"

#include <iostream>

namespace eps {

struct GlobalReducePlan
{
    int *mapping;       /*[num_tokens, ep_world_size]*/
    int *inclusive_sum; /*[ep_world_size, num_stages]*/
};

struct GlobalReducePlanKernelParams
{
    const int *expert_indices;   /*[num_tokens, topk]*/
    int *tailing_experts;        /*[ep_world_size, num_tokens]*/
    int *sorted_tailing_experts; /*[ep_world_size, num_tokens]*/
    GlobalReducePlan reduce_plan;
    ExpandForCommPlan expand_plan;
    int num_tokens;
    int num_experts;
    int num_stages;
    int topk;
    int ep_world_size;
};

template<typename T>
struct GlobalReduceKernelParams {
    GlobalReducePlan reduce_plan;
    int num_tokens;
    int ep_world_size;
    int64_t hidden_size;
    const T* src;
    T* dst;
    const T* shared;
};

void getGlobalReducePlan(GlobalReducePlanKernelParams p, cudaStream_t stream);

void getGlobalReducePlan_ver2(GlobalReducePlanKernelParams p, cudaStream_t stream);

template<typename T>
void globalReduce(GlobalReduceKernelParams<T> p, cudaStream_t stream);

class GlobalReducer {
 public:
  struct Params {
    int num_tokens;
    int num_experts;
    int num_stages;
    int topk;
    int ep_world_size;
    int64_t hidden_size;
  };

  GlobalReducer() = default;

  GlobalReducer(Params p) : p_{p} {}

  size_t getWorkspaceSize() {
    int mapping = p_.num_tokens * p_.ep_world_size;
    int inclusive_sum = p_.ep_world_size * p_.num_stages;
    int tailing_experts = p_.ep_world_size * p_.num_tokens;
    int sorted_tailing_experts = p_.ep_world_size * p_.num_tokens;

    size_t bytes = sizeof(int) * (mapping + inclusive_sum + tailing_experts + sorted_tailing_experts);
    return bytes;
  }

  GlobalReducePlan plan(const int *expert_indices, ExpandForCommPlan expand_plan, void* ws, cudaStream_t stream) {
    int* mapping = (int*)ws;
    int* inclusive_sum = mapping + p_.num_tokens * p_.ep_world_size;
    int* tailing_experts = inclusive_sum + p_.ep_world_size * p_.num_stages;
    int* sorted_tailing_experts = tailing_experts + p_.ep_world_size * p_.num_tokens;

    GlobalReducePlan reduce_plan{
        .mapping = mapping,
        .inclusive_sum = inclusive_sum
    };

    GlobalReducePlanKernelParams p{
        .expert_indices = expert_indices,
        .tailing_experts = tailing_experts,
        .sorted_tailing_experts = sorted_tailing_experts,
        .reduce_plan = reduce_plan,
        .expand_plan = expand_plan,
        .num_tokens = p_.num_tokens,
        .num_experts = p_.num_experts,
        .num_stages = p_.num_stages,
        .topk = p_.topk,
        .ep_world_size = p_.ep_world_size
    };

    getGlobalReducePlan_ver2(p, stream);
    // getGlobalReducePlan(p, stream);

    return reduce_plan;
  }

  template <typename T>
  void run(const T* src, T* dst, GlobalReducePlan reduce_plan, const T* shared, cudaStream_t stream) {
    GlobalReduceKernelParams<T> p{
        .reduce_plan = reduce_plan,
        .num_tokens = p_.num_tokens,
        .ep_world_size = p_.ep_world_size,
        .hidden_size = p_.hidden_size,
        .src = src,
        .dst = dst,
        .shared = shared
    };

    globalReduce(p, stream);
  }

 private:
  Params p_{};
};

}