#include "src/scheduler/comm_expander.cuh"
#include "src/scheduler/utils.cuh"

namespace eps {

struct GetExpandForCommPlanHelper {
 public:
  static constexpr int MAX_NUM_THREADS = 128;
  static constexpr int LEAST_NUM_TOKENS_PER_THREAD = 16;
  GetExpandForCommPlanHelper(ExpandForCommPlanKernelParams p) {
    params_ = p;
    num_stages_ = p.num_stages;
    num_experts_ = p.num_experts;
    ep_size_ = p.ep_world_size;
    num_threads_ = std::min(CEILDIV((size_t)p.num_tokens, LEAST_NUM_TOKENS_PER_THREAD), (size_t)MAX_NUM_THREADS);
    // for num_total_tokens_ == 0
    num_threads_ = std::max(num_threads_, 1);
    const int num_experts_per_device = num_experts_ / ep_size_;
    shared_memory_bytes_ = sizeof(int) * (num_threads_ + 1) * ep_size_ * num_experts_per_device;
  }
  ExpandForCommPlanKernelParams params_;
  int32_t num_stages_;
  int32_t num_experts_;
  int32_t ep_size_;
  int32_t num_threads_;
  size_t shared_memory_bytes_;
  void* debug_ws_{nullptr};
};

__global__ void getExpandForCommPlanKernel_ver2(GetExpandForCommPlanHelper h, ExpandForCommPlanKernelParams p) {
  const int NUM_THREADS = blockDim.x;
  // ========params========
  const int NUM_EXPERTS = p.num_experts;
  const int NUM_STAGES = p.num_stages;
  const int DP_NUM_TOKENS = p.num_tokens;
  const int TOPK = p.topk;
  const int WORLD_SIZE = p.ep_world_size;
  assert(NUM_EXPERTS % WORLD_SIZE == 0);
  const int NUM_EXPERTS_PER_DEVICE = NUM_EXPERTS / WORLD_SIZE;
  assert(NUM_EXPERTS_PER_DEVICE % p.num_stages == 0);
  const int NUM_EXPERTS_PER_STAGE = NUM_EXPERTS_PER_DEVICE / p.num_stages;
  const int tid = threadIdx.x;
  // ========arrage========
  // on GMEM
  CREATE_EMPTY_STRUCT(__token_ep_leading_experts);
  TensorView<int, 2, __token_ep_leading_experts> leading_experts(p.sorted_leading_experts, DP_NUM_TOKENS, WORLD_SIZE);

  CREATE_EMPTY_STRUCT(__topk_expert_ids);
  TensorView<const int, 2, __topk_expert_ids> expert_indices(p.expert_indices, DP_NUM_TOKENS, TOPK);

  // on SMEM
  extern __shared__ int smem[];
  CREATE_EMPTY_STRUCT(__s_threads_leading_expert_inclusive_cumsums);
  TensorView<int, 3, __s_threads_leading_expert_inclusive_cumsums> inclusive_sums(smem, NUM_THREADS + 1, WORLD_SIZE, NUM_EXPERTS_PER_DEVICE);

  // ========Calculate the Leading expert on EACH device for EACH token========
  const int NO_LEAD = -993;

  // init
  for (int src_idx = tid; src_idx < DP_NUM_TOKENS; src_idx += NUM_THREADS) {
    for (int ep_idx = 0; ep_idx < WORLD_SIZE; ++ep_idx) {
      leading_experts.at(src_idx, ep_idx) = NO_LEAD;
    }
  }

  // calc leading expert for each device
  for (int src_idx = tid; src_idx < DP_NUM_TOKENS; src_idx += NUM_THREADS) {
    for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
      int expert_id = expert_indices.at(src_idx, k_idx);
      // skip zero-computation experts
      if (expert_id < 0) continue;
      int target_ep_idx = expert_id / NUM_EXPERTS_PER_DEVICE;
      int cur_lead_idx = leading_experts.at(src_idx, target_ep_idx);

      int expert_idx = expert_id % NUM_EXPERTS_PER_DEVICE;
      if (cur_lead_idx == NO_LEAD || expert_idx < cur_lead_idx) {
        leading_experts.at(src_idx, target_ep_idx) = expert_idx;
      }
    }
  }

  // ========Accumulate the counts of routed tokens within the responsibility of each THREAD for each EXPERT========
  // set all leading experts of ep,stage to 0
  for (int target_ep_idx = 0; target_ep_idx < WORLD_SIZE; ++target_ep_idx) {
    for (int expert_idx = 0; expert_idx < NUM_EXPERTS_PER_DEVICE; ++expert_idx) {
      if (tid == 0) {
        inclusive_sums.at(0, target_ep_idx, expert_idx) = 0;
      }
      inclusive_sums.at(tid + 1, target_ep_idx, expert_idx) = 0;
    }
  }
  __syncthreads();

  // for EACH device, query the leanding expert, just inc the CORRESPONDING count.
  // EACH thread only query its responsible tokens
  for (int src_idx = tid; src_idx < DP_NUM_TOKENS; src_idx += NUM_THREADS) {
    for (int target_ep_idx = 0; target_ep_idx < WORLD_SIZE; ++target_ep_idx) {
      int expert_idx = leading_experts.at(src_idx, target_ep_idx);
      if (expert_idx != NO_LEAD) {
        // WARNING: We put in counts first, which will change to cumsums later.
        // NOTE: We write to tid + 1 row for exclusive cumsums, starting from zero.
        inclusive_sums.at(tid + 1, target_ep_idx, expert_idx) += 1;
      }
    }
  }
  __syncthreads();

  // ========Get the exclusive cumsums of leading expert for each THREAD and for EACH target ep idx========
  for (int target_ep_idx = 0; target_ep_idx < WORLD_SIZE; ++target_ep_idx) {
    // NOTE: Each thread temporarily work on columns to perform independent parallel reductions
    for (int expert_idx = tid; expert_idx < NUM_EXPERTS_PER_DEVICE; expert_idx += NUM_THREADS) {
      for (int row_idx = 1; row_idx <= NUM_THREADS; ++row_idx) {
        inclusive_sums.at(row_idx, target_ep_idx, expert_idx) += inclusive_sums.at(row_idx - 1, target_ep_idx, expert_idx);
      }
    }
  }
  __syncthreads();

  // ========Inclusive Cumsum========
  // the flattened array length is NUM_EXPERTS * ep_size
  // since NUM_THREADS >= NUM_EXPERTS, ep_size is relatively small.
  // little work, one thread is enough.
  if (tid == 0) {
    int ep_stage_num_tails;
    int prev = 0;
    for (int target_ep_idx = 0; target_ep_idx < WORLD_SIZE; ++target_ep_idx) {
      for (int stage_idx = 0; stage_idx < NUM_STAGES; ++stage_idx) {
        int cur_idx = target_ep_idx * NUM_STAGES + stage_idx;
        // count the tail of this stage.
        ep_stage_num_tails = 0;
        for (int expert_idx = stage_idx * NUM_EXPERTS_PER_STAGE; expert_idx < (stage_idx + 1) * NUM_EXPERTS_PER_STAGE; ++expert_idx) {
          ep_stage_num_tails += inclusive_sums.at(NUM_THREADS, target_ep_idx, expert_idx);
        }
        p.expand_plan.inclusive_sum[cur_idx] = prev + ep_stage_num_tails;
        prev = p.expand_plan.inclusive_sum[cur_idx];
      }
    }
  }
  __syncthreads();

  // ========INPLACE Exclusive cumsum of ep leading expert========
  if (tid == 0) {
    int sum = 0;
    int cur;
    int* arr = &inclusive_sums.at(NUM_THREADS, 0, 0);
    for (int idx = 0; idx < NUM_EXPERTS_PER_DEVICE * WORLD_SIZE; ++idx) {
      cur = arr[idx];
      arr[idx] = sum;
      sum += cur;
    }
  }
  __syncthreads();

  // ========Get the mapping from send idx to source marker========
  // source marker: denotes the source index of the token and the ep idx it will be sent to.
  // leading_experts <= NUM_EXPERTS, hence we can use Counting Sort.

  for (int src_idx = tid; src_idx < DP_NUM_TOKENS; src_idx += NUM_THREADS) {
    // for each token, check all of its leading expert on EACH device
    for (int target_ep_idx = 0; target_ep_idx < WORLD_SIZE; ++target_ep_idx) {
      const int target_ep_lead_idx = leading_experts.at(src_idx, target_ep_idx);
      if (target_ep_lead_idx != NO_LEAD) {
        const int src_marker = target_ep_idx * DP_NUM_TOKENS + src_idx;
        const int EP_EXPERT_BASE = inclusive_sums.at(NUM_THREADS, target_ep_idx, target_ep_lead_idx);
        // t_ means current thread
        int& t_expert_offset = inclusive_sums.at(tid, target_ep_idx, target_ep_lead_idx);
        p.expand_plan.mapping[EP_EXPERT_BASE + t_expert_offset] = src_marker;
        t_expert_offset++;
      }
    }
  }
}

void getExpandForCommPlan_ver2(ExpandForCommPlanKernelParams p, cudaStream_t stream) {
  if (p.num_tokens == 0) {
    cudaMemsetAsync(p.expand_plan.inclusive_sum, 0, sizeof(int) * p.ep_world_size * p.num_stages, stream);
    return;
  }
  GetExpandForCommPlanHelper h(p);
  const int SHARED_MEM_BYTES = h.shared_memory_bytes_;
  cudaFuncSetAttribute(getExpandForCommPlanKernel_ver2, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM_BYTES);
  getExpandForCommPlanKernel_ver2<<<1, h.num_threads_, SHARED_MEM_BYTES, stream>>>(h, p);
}

}  // namespace eps
