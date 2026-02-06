#include "src/scheduler/gemm_expander.cuh"
#include "src/scheduler/utils.cuh"

namespace eps {

struct ExpandForGemmPlanKernelHelper {
  static constexpr int MAX_NUM_THREADS = 128;
  static constexpr int LEAST_NUM_TOKENS_PER_THREAD = 16;

  ExpandForGemmPlanKernelHelper(ExpandForGemmPlanKernelParams p) : p_{p} {
    // Actually we do not need to process total_num_tokens_.
    // The actual num of tokens is the total num of recved tokens of every device.
    // But recv_plan is on device. So we use the upper bound of the recved tokens.
    total_num_tokens_ = p.max_num_tokens_per_gpu * p.ep_world_size;
    num_local_experts_ = (p_.local_expert_end - p_.local_expert_begin) + 1;
    num_threads_ = std::min(CEILDIV(total_num_tokens_, LEAST_NUM_TOKENS_PER_THREAD), (size_t)MAX_NUM_THREADS);

    // num_threads_ <= MAX_NUM_THREADS. You can set this larger.
    // But you want to ensure that num_local_experts_ * num_threads_ is not too much.
    // You do not want to make the lauching of blocks delayed due to the massive need of shared memory.
    shared_memory_bytes_ = num_local_experts_ * (num_threads_ + 1) * sizeof(int);
  }

  ExpandForGemmPlanKernelParams p_;
  size_t total_num_tokens_;
  size_t num_local_experts_;
  size_t num_threads_;
  size_t shared_memory_bytes_;
};

__global__ void getExpandForGemmPlanKernel_ver2(ExpandForGemmPlanKernelParams p) {
  // ========params========
  const int tid = threadIdx.x;
  const int NUM_THREADS = blockDim.x;
  const int NUM_RECVED_TOKENS = p.inclusive_sum[p.ep_world_size * p.num_stages - 1];
  const int TOPK = p.topk;

  // LOCAL_EXPERT_BEGIN <= expert
  const int LOCAL_EXPERT_BEGIN = p.local_expert_begin;
  // expert < LOCAL_EXPERT_END
  const int LOCAL_EXPERT_END = p.local_expert_end;
  const int NUM_LOCAL_EXPERTS = LOCAL_EXPERT_END - LOCAL_EXPERT_BEGIN;
  // ========tensors========
  CREATE_EMPTY_STRUCT(__RECVED_TOKEN_TOPK_EXPERT_IDS);
  TensorView<int, 2, __RECVED_TOKEN_TOPK_EXPERT_IDS> recved_token_topk_expert_ids(p.expert_indices, NUM_RECVED_TOKENS, TOPK);

  CREATE_EMPTY_STRUCT(__THREAD_EXPERT_HIT_COUNTS);
  extern __shared__ int total_smem[];
  // the first row is for computing the exclusive cumsum
  TensorView<int, 2, __THREAD_EXPERT_HIT_COUNTS> exclusive_sums(total_smem, NUM_THREADS + 1, NUM_LOCAL_EXPERTS);

  CREATE_EMPTY_STRUCT(__O_LOCAL_EXPERT_HIT_COUNTS);
  // o_ means output
  TensorView<int, 1, __O_LOCAL_EXPERT_HIT_COUNTS> o_exclusive_sums(p.expand_plan.exclusive_sum, NUM_LOCAL_EXPERTS + 1);

  CREATE_EMPTY_STRUCT(__O_SORTED_IDX_TO_DUP_MARKERS);
  // o_ means output
  // the allocated space is more than NUM_RECVED_TOKENS * TOPK.
  // the total hits is even smaller than NUM_RECVED_TOKENS * TOPK.
  TensorView<int, 1, __O_SORTED_IDX_TO_DUP_MARKERS> o_dup_idx_to_recved_markers(p.expand_plan.mapping, NUM_RECVED_TOKENS * TOPK);

  // ========1. Count the num of tokens routed to certain expert in a thread========
  // memset
  for (int local_expert_idx = 0; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
    exclusive_sums.at(tid, local_expert_idx) = 0;
  }
  if (tid == 0) {
    // set the last row.
    for (int local_expert_idx = 0; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
      exclusive_sums.at(NUM_THREADS, local_expert_idx) = 0;
    }
  }
  __syncthreads();

  // Here, each THREAD is responsible for processing several tokens(actually their topk expert ids, all gathered from all devices)
  // THREAD will record the hits of each expert in SMEM.
  for (int recved_token_idx = tid; recved_token_idx < NUM_RECVED_TOKENS; recved_token_idx += NUM_THREADS) {
    for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
      int global_expert_id = recved_token_topk_expert_ids.at(recved_token_idx, k_idx);
      if (LOCAL_EXPERT_BEGIN <= global_expert_id && global_expert_id < LOCAL_EXPERT_END) {
        // expert of current ep device
        int local_expert_idx = global_expert_id - LOCAL_EXPERT_BEGIN;
        // NOTE: each thread write to tid+1. Because in the exclusive sum, when traversing to the tid's row, we just need its preceding rows.
        exclusive_sums.at(tid + 1, local_expert_idx) += 1;
      }
    }
  }
  __syncthreads();

  // ========2. Calc cumsum inplace. Col-wise========
  for (int col_idx = tid; col_idx < NUM_LOCAL_EXPERTS; col_idx += NUM_THREADS) {
    // NOTE: Here, each THREAD temporarily operates on a col
    // dims are indenpendent, so each THREAD can work on its own to calc the cumsum of a single col.
    // NOTE: less than or equal to
    for (int row_idx = 1; row_idx <= NUM_THREADS; ++row_idx) {
      // e.g. say there are 3 threads.
      // each thread contributes to the hit counts as
      // thread 0: 3,2
      // thread 1: 4,1
      // thread 2: 0,2
      // exclusive cumsums will be
      // row 0[for thread 0]: 0,0
      // row 1[for thread 1]: 3,2
      // row 2[for thread 2]: 7,3
      // row 3[for global]:   7,5
      exclusive_sums.at(row_idx, col_idx) += exclusive_sums.at(row_idx - 1, col_idx);
    }
  }
  __syncthreads();

  if (tid == 0) {
    o_exclusive_sums.at(0) = 0;
    int prev = 0;
    int cur = 0;
    for (int local_expert_idx = 0; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
      cur = exclusive_sums.at(NUM_THREADS, local_expert_idx);
      // calc last row exclusive cumsums to the output.
      o_exclusive_sums.at(local_expert_idx + 1) = prev + cur;
      prev = prev + cur;
    }
  }
  __syncthreads();

  // ========3. Calc global expert offsets.========
  CREATE_EMPTY_STRUCT(S_EXPERT_OFFSETS);
  // WARNING: From now on, exclusive_sums[NUM_THREADS] will be inclusive sum;
  TensorView<int, 1, S_EXPERT_OFFSETS> s_expert_offsets(&exclusive_sums.at(NUM_THREADS, 0), NUM_LOCAL_EXPERTS);
  if (tid == 0) {
    int prev = s_expert_offsets.at(0);
    int temp;
    s_expert_offsets.at(0) = 0;
    for (int local_expert_idx = 1; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
      // e.g.
      // if total hits of expert 4-7 are 3,2,1,4
      // the offsets is the exclusive cumsums 0,3,5,6
      temp = s_expert_offsets.at(local_expert_idx);
      s_expert_offsets.at(local_expert_idx) = prev + s_expert_offsets.at(local_expert_idx - 1);
      prev = temp;
    }
  }
  __syncthreads();

  // ========4. Register the duplication markers with each expert's offset========
  // if expert 4-7 is on current ep device. And the recved tokens has topk expert ids as
  // tok0: [1,4,7,3] tok1: [4,6,7,5], tok2: [5,2,3,0]
  // duplicatation marker is (recved_token_idx, expert_topk_idx), through which we perform local reduce.
  // we traverse all 3 tokens received, and register their markers for expert 4-7
  // expert 4: (0, 3), (1, 0)
  // expert 5: (1, 3), (2, 0). For topk=4, (1, 3) will become (1*4+3 = 7) as marker
  // expert 6: (1, 1)
  // expert 7: (0, 2), (1, 2)

  CREATE_EMPTY_STRUCT(__THREADS_EXPERT_OFFSETS);
  // create another view. Just to remind that this offset is changing.
  TensorView<int, 1, __THREADS_EXPERT_OFFSETS> s_cur_thread_expert_offsets(&exclusive_sums.at(tid, 0), NUM_LOCAL_EXPERTS);

  for (int recved_token_idx = tid; recved_token_idx < NUM_RECVED_TOKENS; recved_token_idx += NUM_THREADS) {
    for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
      // NOTE: recved tokens are already arrange according to their leading experts.
      const int recved_marker = recved_token_idx * TOPK + k_idx;
      const int global_expert_id = recved_token_topk_expert_ids.at(recved_token_idx, k_idx);
      if (LOCAL_EXPERT_BEGIN <= global_expert_id && global_expert_id < LOCAL_EXPERT_END) {
        // expert of current ep device
        const int local_expert_idx = global_expert_id - LOCAL_EXPERT_BEGIN;
        // get the offset of THIS LOCAL EXPERT of current THREAD
        const int EXPERT_BASE = s_expert_offsets.at(local_expert_idx);
        // ref to the offset
        int& cur_thread_this_expert_offset = s_cur_thread_expert_offsets.at(local_expert_idx);

        const int offset = EXPERT_BASE + cur_thread_this_expert_offset;
        o_dup_idx_to_recved_markers.at(offset) = recved_marker;

        // NOTE: increment the offset.
        cur_thread_this_expert_offset += 1;
      }
    }
  }
}

void getExpandForGemmPlan_ver2(ExpandForGemmPlanKernelParams p, cudaStream_t stream) {
  ExpandForGemmPlanKernelHelper helper(p);
  cudaFuncSetAttribute(getExpandForGemmPlanKernel_ver2, cudaFuncAttributeMaxDynamicSharedMemorySize, helper.shared_memory_bytes_);

  getExpandForGemmPlanKernel_ver2<<<1, helper.num_threads_, helper.shared_memory_bytes_, stream>>>(p);
}

}  // namespace eps