#include <cub/cub.cuh>

#include "src/scheduler/common.cuh"
#include "src/scheduler/global_reducer.cuh"
#include "src/scheduler/utils.cuh"

namespace eps {

struct GlobalReducePlanKernelHelper {
  static constexpr int MAX_NUM_THREADS = 512;
  static constexpr int LEAST_NUM_TOKENS_PER_THREAD = 16;

 public:
  GlobalReducePlanKernelHelper(GlobalReducePlanKernelParams p) : p_{p} {
    num_total_tokens_ = p.ep_world_size * p.num_tokens;
    num_threads_ = std::min(CEILDIV(num_total_tokens_, LEAST_NUM_TOKENS_PER_THREAD), (size_t)MAX_NUM_THREADS);
    // for num_total_tokens_ == 0
    num_threads_ = std::max(num_threads_, 1UL);
    assert(p_.num_experts % p_.ep_world_size == 0);
    num_experts_per_device_ = p_.num_experts / p_.ep_world_size;
    shared_memory_bytes_ = sizeof(int) * (s_threads_tail_counts_numel());
  }
  __host__ __device__ size_t s_threads_tail_counts_numel() { return (num_threads_ + 1) * num_experts_per_device_; }
  GlobalReducePlanKernelParams p_;
  size_t num_threads_;
  size_t num_total_tokens_;
  size_t num_experts_per_device_;
  size_t shared_memory_bytes_;
};

__global__ void getGlobalReducePlanKernel_ver2(GlobalReducePlanKernelParams p) {
  // ========params========
  const int WORLD_SIZE = p.ep_world_size;
  const int NUM_STAGES = p.num_stages;
  const int NUM_DP_TOKENS = p.num_tokens;
  const int TOPK = p.topk;
  assert(p.num_experts % p.ep_world_size == 0);
  const int NUM_EXPERTS_PER_DEVICE = p.num_experts / p.ep_world_size;
  assert(NUM_EXPERTS_PER_DEVICE % NUM_STAGES == 0);
  const int NUM_EXPERTS_PER_STAGE = NUM_EXPERTS_PER_DEVICE / NUM_STAGES;
  const int NUM_TOTAL_SENT_TOKENS = p.expand_plan.inclusive_sum[WORLD_SIZE * NUM_STAGES - 1];

  const int tid = threadIdx.x;
  const int NUM_THREADS = blockDim.x;
  // ========tensors========
  CREATE_EMPTY_STRUCT(__src_expert_ids);
  TensorView<const int, 2, __src_expert_ids> src_expert_ids(p.expert_indices, NUM_DP_TOKENS, TOPK);

  // TensorView<int, 2> target_ep_sorted_tails(p.target_ep_sorted_tailing_experts_, WORLD_SIZE, NUM_EXPERTS_PER_DEVICE);

  CREATE_EMPTY_STRUCT(__send_idx_to_ep_tail_experts);
  TensorView<int, 1, __send_idx_to_ep_tail_experts> send_idx_to_tail_idx(p.tailing_experts, NUM_TOTAL_SENT_TOKENS);

  CREATE_EMPTY_STRUCT(__ep_stage_send_inclusive_cumcums);
  TensorView<int, 2, __ep_stage_send_inclusive_cumcums> comm_inclusive_sum(p.expand_plan.inclusive_sum, WORLD_SIZE, NUM_STAGES);

  CREATE_EMPTY_STRUCT(__send_idx_to_src_marker);
  TensorView<const int, 1, __send_idx_to_src_marker> send_idx_to_src_marker(p.expand_plan.mapping, NUM_TOTAL_SENT_TOKENS);

  extern __shared__ int smem[];

  CREATE_EMPTY_STRUCT(__s_threads_ep_tail_idx_exclusive_cumsums);
  TensorView<int, 2, __s_threads_ep_tail_idx_exclusive_cumsums> ep_tail_exclusive_sums(smem, NUM_THREADS + 1, NUM_EXPERTS_PER_DEVICE);

  CREATE_EMPTY_STRUCT(__o_ep_stage_collect_idx_inclusive_cumsums);
  TensorView<int, 2, __o_ep_stage_collect_idx_inclusive_cumsums> reduce_inclusive_sum(p.reduce_plan.inclusive_sum, WORLD_SIZE, NUM_STAGES);

  CREATE_EMPTY_STRUCT(__o_src_marker_to_collect_idx);
  TensorView<int, 1, __o_src_marker_to_collect_idx> src_marker_to_collect_idx(p.reduce_plan.mapping, NUM_DP_TOKENS * WORLD_SIZE);

  // memset
  for (int src_idx = tid; src_idx < NUM_DP_TOKENS; src_idx += NUM_THREADS) {
    for (int ep_idx = 0; ep_idx < WORLD_SIZE; ++ep_idx) {
      int src_marker = ep_idx * NUM_DP_TOKENS + src_idx;
      // HACK: INVALID id
      src_marker_to_collect_idx.at(src_marker) = NUM_DP_TOKENS * WORLD_SIZE;
    }
  }
  __syncthreads();

  // ========Get the tailing expert for each target EP device========
  // send idxes are sorted by target ep idx.
  // tailing expert is device-wise.
  for (int target_ep_idx = 0; target_ep_idx < WORLD_SIZE; ++target_ep_idx) {
    const int target_ep_send_idx_begin = target_ep_idx == 0 ? 0 : comm_inclusive_sum.at(target_ep_idx - 1, NUM_STAGES - 1);
    const int target_ep_send_idx_end = comm_inclusive_sum.at(target_ep_idx, NUM_STAGES - 1);
    const int num_total_tokens_to_target_ep = target_ep_send_idx_end - target_ep_send_idx_begin;

    const int target_ep_expert_begin = target_ep_idx * NUM_EXPERTS_PER_DEVICE;
    // we do not use closed right bound because many of the bounds here are open on the right.

    const int _num_tokens_per_thread = CEILDIV(num_total_tokens_to_target_ep, NUM_THREADS);
    const int _t_send_idx_offset_bound = std::min(num_total_tokens_to_target_ep, (tid + 1) * _num_tokens_per_thread);

    // init
    for (int expert_idx = 0; expert_idx < NUM_EXPERTS_PER_DEVICE; ++expert_idx) {
      ep_tail_exclusive_sums.at(tid + 1, expert_idx) = 0;
      if (tid == 0) {
        ep_tail_exclusive_sums.at(0, expert_idx) = 0;
      }
    }
    __syncthreads();

    // extract the tailing expert for current ep
    // for each sent token
    // NOTE: Must use STABLE SORT tail expert id w.r.t. send_idx
    // NOTE: So each thread MUST operate sequentially on continous send idxes
    // Continuous traversing manner.
    for (int _send_idx_offset = tid * _num_tokens_per_thread; _send_idx_offset < _t_send_idx_offset_bound; ++_send_idx_offset) {
      const int send_idx = target_ep_send_idx_begin + _send_idx_offset;
      // src_marker: [ep, src_idx]
      const int src_marker = send_idx_to_src_marker.at(send_idx);
      const int src_idx = src_marker % NUM_DP_TOKENS;
      // HACK: INVALID TAIL
      int tailing_expert_idx = -1;

      for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
        int _expert_id = src_expert_ids.at(src_idx, k_idx);
        // skip zero-computation experts. run reduce separately
        if (_expert_id < 0) continue;
        int ep_expert_idx = _expert_id - target_ep_expert_begin;
        if (0 <= ep_expert_idx && ep_expert_idx < NUM_EXPERTS_PER_DEVICE && ep_expert_idx > tailing_expert_idx) {
          tailing_expert_idx = ep_expert_idx;
        }
      }

      send_idx_to_tail_idx.at(send_idx) = tailing_expert_idx;
      // WARNING: Count the hits now. not cumsums yet.
      ep_tail_exclusive_sums.at(tid + 1, tailing_expert_idx) += 1;
    }
    __syncthreads();

    // ========Exclusive rows cumsum========
    // NOTE: Each Thread's role temporarily changes to columns
    for (int expert_idx = tid; expert_idx < NUM_EXPERTS_PER_DEVICE; expert_idx += NUM_THREADS) {
      for (int row_idx = 1; row_idx <= NUM_THREADS; ++row_idx) {
        ep_tail_exclusive_sums.at(row_idx, expert_idx) += ep_tail_exclusive_sums.at(row_idx - 1, expert_idx);
      }
    }
    __syncthreads();

    // ========Per expert -> stage counts========
    // little work. only 1 thread is enough
    if (tid == 0) {
      for (int stage_idx = 0; stage_idx < NUM_STAGES; ++stage_idx) {
        reduce_inclusive_sum.at(target_ep_idx, stage_idx) = 0;
        for (int expert_idx = stage_idx * NUM_EXPERTS_PER_STAGE; expert_idx < (stage_idx + 1) * NUM_EXPERTS_PER_STAGE; ++expert_idx) {
          reduce_inclusive_sum.at(target_ep_idx, stage_idx) += ep_tail_exclusive_sums.at(NUM_THREADS, expert_idx);
        }
      }
    }
    __syncthreads();

    // ========Calc offset for each tail expert========
    // Counts to Exclusive sum
    // little work. only 1 thread.
    if (tid == 0) {
      int sum = 0;
      int cur;
      for (int expert_idx = 0; expert_idx < NUM_EXPERTS_PER_DEVICE; ++expert_idx) {
        cur = ep_tail_exclusive_sums.at(NUM_THREADS, expert_idx);
        ep_tail_exclusive_sums.at(NUM_THREADS, expert_idx) = sum;
        sum += cur;
      }
      assert(sum == num_total_tokens_to_target_ep);
    }
    __syncthreads();

    // ========Counting Sort========
    // Key: tail expert
    // Value: send idx -> src marker

    // NOTE: Stable Sort
    // expert cumsums have already calculated from the Continuous traversing manner.
    for (int _send_idx_offset = tid * _num_tokens_per_thread; _send_idx_offset < _t_send_idx_offset_bound; ++_send_idx_offset) {
      const int send_idx = target_ep_send_idx_begin + _send_idx_offset;
      // int src_marker = target_ep_idx * DP_NUM_TOKENS + src_idx;
      const int send_src_marker = send_idx_to_src_marker.at(send_idx);
      int src_idx = send_src_marker % NUM_DP_TOKENS;
      // WARNING: send_src_marker have different encoding pattern with collect_src_marker
      int collect_src_marker = src_idx * WORLD_SIZE + target_ep_idx;

      const int tail_expert_idx = send_idx_to_tail_idx.at(send_idx);
      if (0 <= tail_expert_idx & tail_expert_idx < NUM_EXPERTS_PER_DEVICE) {
        const int EXPERT_BASE = ep_tail_exclusive_sums.at(NUM_THREADS, tail_expert_idx);
        int &t_expert_offset = ep_tail_exclusive_sums.at(tid, tail_expert_idx);
        int collect_idx = target_ep_send_idx_begin + EXPERT_BASE + t_expert_offset;

        src_marker_to_collect_idx.at(collect_src_marker) = collect_idx;

        t_expert_offset++;
      }
    }
    __syncthreads();
  }

  // ========Ep stage count to inclusive cumsums========
  if (tid == 0) {
    for (int i = 1; i < WORLD_SIZE * NUM_STAGES; ++i) {
      p.reduce_plan.inclusive_sum[i] += p.reduce_plan.inclusive_sum[i - 1];
    }
  }
  __syncthreads();
}

void getGlobalReducePlan_ver2(GlobalReducePlanKernelParams p, cudaStream_t stream) {
  if (p.num_tokens == 0) {
    cudaMemsetAsync(p.reduce_plan.inclusive_sum, 0, sizeof(int) * p.ep_world_size * p.num_stages, stream);
    return;
  }
  GlobalReducePlanKernelHelper helper(p);
  cudaFuncSetAttribute(getGlobalReducePlanKernel_ver2, cudaFuncAttributeMaxDynamicSharedMemorySize, helper.shared_memory_bytes_);
  getGlobalReducePlanKernel_ver2<<<1, helper.num_threads_, helper.shared_memory_bytes_, stream>>>(p);
}

}  // namespace eps
