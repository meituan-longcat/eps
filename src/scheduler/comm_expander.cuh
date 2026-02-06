#pragma once

namespace eps {

struct ExpandForCommPlan {
  int* mapping; /*[ep_world_size * num_tokens]*/
  int* inclusive_sum; /*[ep_world_size * num_stages]*/
};

struct ExpandForCommPlanKernelParams {
  const int* expert_indices;   /* [num_tokens, topk] */
  int* sorted_leading_experts; /*[ep_world_size, num_tokens]*/
  ExpandForCommPlan expand_plan;
  int num_tokens;
  int topk;
  int num_experts;
  int num_stages;
  int ep_world_size;
  unsigned long long int* tile_states; /*ep_world_size*/
};

template <typename T>
struct ExpandForCommKernelParams {
  ExpandForCommPlan expand_plan;
  int num_tokens;
  int topk;
  int num_stages;
  int ep_world_size;
  int cols;
  const T* src;
  T* dst;
};

void getExpandForCommPlan(ExpandForCommPlanKernelParams p, cudaStream_t stream);

void getExpandForCommPlan_ver2(ExpandForCommPlanKernelParams p, cudaStream_t stream);

template <typename T>
void expandForComm(ExpandForCommKernelParams<T> p, cudaStream_t stream);

class CommExpander {
public:
  struct Params {
    int num_tokens;
    int topk;
    int num_experts;
    int num_stages;
    int ep_world_size;
  };

  CommExpander() = default;

  CommExpander(Params p) : p_{p} {}

  size_t getWorkspaceSize() {
    size_t mapping = p_.ep_world_size * p_.num_tokens;
    size_t sorted_leading_experts = p_.ep_world_size * p_.num_tokens;
    size_t tile_states = p_.ep_world_size;
    size_t bytes = (mapping + sorted_leading_experts) * sizeof(int) + tile_states * sizeof(unsigned long long int);
    return bytes;
  }

  ExpandForCommPlan plan(const int* expert_indices, int* inclusive_sum, void* ws, cudaStream_t stream) {
    int* mapping = (int*)ws;
    int* sorted_leading_experts = mapping + p_.ep_world_size * p_.num_tokens;
    auto tile_states = (unsigned long long int*)(sorted_leading_experts + p_.ep_world_size * p_.num_tokens);
    cudaMemsetAsync(tile_states, 0, p_.ep_world_size * sizeof(unsigned long long int), stream);

    ExpandForCommPlan expand_plan{
        .mapping = mapping,
        .inclusive_sum = inclusive_sum,
    };

    ExpandForCommPlanKernelParams p{
        .expert_indices = expert_indices,
        .sorted_leading_experts = sorted_leading_experts,
        .expand_plan = expand_plan,
        .num_tokens = p_.num_tokens,
        .topk = p_.topk,
        .num_experts = p_.num_experts,
        .num_stages = p_.num_stages,
        .ep_world_size = p_.ep_world_size,
        .tile_states = tile_states};
    getExpandForCommPlan_ver2(p, stream);
    return expand_plan;
  }

  template <typename T>
  void run(const T* src, T* dst, int cols, ExpandForCommPlan expand_plan, cudaStream_t stream) {
    ExpandForCommKernelParams<T> p{
        .expand_plan = expand_plan,
        .num_tokens = p_.num_tokens,
        .topk = p_.topk,
        .num_stages = p_.num_stages,
        .ep_world_size = p_.ep_world_size,
        .cols = cols,
        .src = src,
        .dst = dst};

    expandForComm(p, stream);
  }

 private:
  Params p_{};
};

} // namespace eps