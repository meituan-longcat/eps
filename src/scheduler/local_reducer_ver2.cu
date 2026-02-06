#include "src/scheduler/local_reducer.cuh"
#include "src/scheduler/utils.cuh"

namespace eps {

struct LocalReducePlanKernelHelper {
  static constexpr int MAX_NUM_TOTAL_DUP_TOKENS = 4096;
  static constexpr int MAX_NUM_THREADS = 1024;
  static constexpr int LEAST_NUM_TOKENS_PER_THREAD = 16;

 public:
  LocalReducePlanKernelHelper(LocalReducePlanKernelParams p) : p_{p} {
    num_total_tokens_ = p_.max_num_tokens_per_gpu * p_.ep_world_size;
    num_threads_ = std::min(CEILDIV(num_total_tokens_, LEAST_NUM_TOKENS_PER_THREAD), (size_t)MAX_NUM_THREADS);
    num_experts_per_device_ = p_.local_expert_end - p_.local_expert_begin;
  }
  // single dp rank
  __host__ __device__ size_t s_threads_tail_counts_numel(size_t num_threads) { return (num_threads + 1) * num_experts_per_device_; }
  LocalReducePlanKernelParams p_;
  size_t num_threads_;
  size_t num_total_tokens_;
  size_t num_experts_per_device_;
};

__global__ void getLocalReducePlanKernel_ver2(LocalReducePlanKernelParams p, LocalReducePlanKernelHelper helper) {
  // ========params========
  const int NUM_THREADS = blockDim.x;
  const int NUM_TOTAL_RECVED_TOKENS = p.inclusive_sum[p.ep_world_size * p.num_stages - 1];

  const int NUM_STAGES = p.num_stages;
  const int LOCAL_EXPERT_BEGIN = p.local_expert_begin;
  const int LOCAL_EXPERT_END = p.local_expert_end;
  const int NUM_LOCAL_EXPERTS = LOCAL_EXPERT_END - LOCAL_EXPERT_BEGIN;
  const int NUM_TOTAL_DUP_TOKENS = p.expand_plan.exclusive_sum[NUM_LOCAL_EXPERTS];
  assert(NUM_LOCAL_EXPERTS % NUM_STAGES == 0);
  const int NUM_EXPERT_PER_STAGE = NUM_LOCAL_EXPERTS / NUM_STAGES;
  const int WORLD_SIZE = p.ep_world_size;
  const int TOPK = p.topk;

  const int tid = threadIdx.x;

  // ========tensors========
  CREATE_EMPTY_STRUCT(__t0);
  TensorView<const int, 2, __t0> expert_indices(p.expert_indices, NUM_TOTAL_RECVED_TOKENS, TOPK);

  CREATE_EMPTY_STRUCT(__t1);
  TensorView<const int, 2, __t1> recv_inclusive_sums(p.inclusive_sum, WORLD_SIZE, NUM_STAGES);

  CREATE_EMPTY_STRUCT(__t2);
  TensorView<int, 1, __t2> recv_idx_to_tail(p.tailing_experts, NUM_TOTAL_RECVED_TOKENS);

  extern __shared__ int smem[];
  CREATE_EMPTY_STRUCT(__t3);
  // s_ means on shared memory
  // exc means exclusive cumsum
  TensorView<int, 2, __t3> tail_exclusive_sums(smem, NUM_THREADS + 1, NUM_LOCAL_EXPERTS);

  CREATE_EMPTY_STRUCT(__t4);
  TensorView<int, 1, __t4> recv_idx_to_reduce_idx(p.sorted_tailing_experts, NUM_TOTAL_RECVED_TOKENS);

  // outputs
  CREATE_EMPTY_STRUCT(__t5);
  TensorView<int, 1, __t5> reduce_idx_to_recved_idx(p.reduce_plan.scale_mapping, NUM_TOTAL_RECVED_TOKENS);

  CREATE_EMPTY_STRUCT(__t6);
  TensorView<int, 2, __t6> reduce_marker_to_dup_idx(p.reduce_plan.mapping, NUM_TOTAL_RECVED_TOKENS, TOPK);

  CREATE_EMPTY_STRUCT(__t7);
  TensorView<int, 2, __t7> reduce_inclusive_sums(p.reduce_plan.inclusive_sum, WORLD_SIZE, NUM_STAGES);

  // NOTE: Reduce Idx indicates the pos of the token when choosing them to perform local reduction.
  // NOTE: Recved Idx indicates the pos of receiving the tokens, related to Leading Experts and the Dispatch Comminication Process

  // ========Process each DP rank, from where the token comes========
  // There are two outcomes that should be processed for each single rank
  // a. DP-stage tailing expert cumcums.
  // At each stage of EPS, we process tokens for several different ranks.
  // Thus, the tailing expert cumsums has 2 dims, recved_from_dp_idx and stage_idx.
  // Meaning that we need to process some tokens for recved_from_dp_idx at stage_idx.
  // We first count the tails of each RANK and STAGE in shared memory, then perform inclusive cumsum of all ranks after the tokens recved from each rank.
  // b. Inner DP Reduce idx to Recved Idx

  // for each DP
  for (int cur_dp_idx = 0; cur_dp_idx < WORLD_SIZE; ++cur_dp_idx) {
    // ========Init========
    // tailing experts
    constexpr int INVALID_TAIL = INT_MAX;
    for (int recv_idx = tid; recv_idx < NUM_TOTAL_RECVED_TOKENS; recv_idx += NUM_THREADS) {
      recv_idx_to_tail.at(recv_idx) = INVALID_TAIL;
    }
    __syncthreads();

    const int dp_recv_idx_start = cur_dp_idx == 0 ? 0 : recv_inclusive_sums.at(cur_dp_idx - 1, NUM_STAGES - 1);
    const int dp_recv_idx_next_start = recv_inclusive_sums.at(cur_dp_idx, NUM_STAGES - 1);
    // WARNING: Do not plus one when using cumsum!
    const int num_total_tokens_cur_dp = dp_recv_idx_next_start - dp_recv_idx_start;
    // traverse all the recved tokens from this dp idx
    // each thread work on several tokens

    // ========Get tailing expert on this device========
    const int num_recv_tokens_per_thread = CEILDIV(num_total_tokens_cur_dp, NUM_THREADS);
    const int recv_tokens_bound = std::min(num_total_tokens_cur_dp, (tid + 1) * num_recv_tokens_per_thread);
    // Each THREAD checks its own portion of tokens
    for (int _recv_idx_offset = tid * num_recv_tokens_per_thread; _recv_idx_offset < recv_tokens_bound; ++_recv_idx_offset) {
      // get the idx by Base + Offset
      int recv_idx = dp_recv_idx_start + _recv_idx_offset;

      for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
        int _expert_id = expert_indices.at(recv_idx, k_idx);
        if (LOCAL_EXPERT_BEGIN <= _expert_id && _expert_id < LOCAL_EXPERT_END) {
          int local_expert_idx = _expert_id - LOCAL_EXPERT_BEGIN;
          int& cur_tail_idx = recv_idx_to_tail.at(recv_idx);
          if (cur_tail_idx == INVALID_TAIL || cur_tail_idx < local_expert_idx) {
            cur_tail_idx = local_expert_idx;
          }
        }
      }
    }
    __syncthreads();
    // ========Count tail on each expert========
    // memset
    for (int local_expert_idx = 0; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
      tail_exclusive_sums.at(tid + 1, local_expert_idx) = 0;
      if (tid == 0) {
        tail_exclusive_sums.at(0, local_expert_idx) = 0;
      }
    }
    __syncthreads();

    for (int _recv_idx_offset = tid * num_recv_tokens_per_thread; _recv_idx_offset < recv_tokens_bound; ++_recv_idx_offset) {
      // get the idx by Base + Offset
      int recv_idx = dp_recv_idx_start + _recv_idx_offset;
      int tail_expert_idx = recv_idx_to_tail.at(recv_idx);
      // NOTE: we write to row of tid+1, for we need the exclusive sum of each expert. It can be used to calc the base start of each EXPERT for each THREAD.
      // WARNING: We put the counts first. Then they will be transformed into cumsums
      if (tail_expert_idx != INVALID_TAIL) {
        tail_exclusive_sums.at(tid + 1, tail_expert_idx) += 1;
      }
    }
    __syncthreads();

    // ========Exclusive Cumsum of rows========
    // NOTE: tid's role temporarily changes. Each THREAD works on several columns to ensure parallelism.
    for (int local_expert_idx = tid; local_expert_idx < NUM_LOCAL_EXPERTS; local_expert_idx += NUM_THREADS) {
      for (int row_idx = 1; row_idx <= NUM_THREADS; ++row_idx) {
        tail_exclusive_sums.at(row_idx, local_expert_idx) += tail_exclusive_sums.at(row_idx - 1, local_expert_idx);
      }
    }
    __syncthreads();

    // ========Write stage tail counts to output========
    // little work. t0 is enough.
    if (tid == 0) {
      // memset
      for (int stage_idx = 0; stage_idx < NUM_STAGES; ++stage_idx) {
        reduce_inclusive_sums.at(cur_dp_idx, stage_idx) = 0;
      }

      for (int local_expert_idx = 0; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
        // WARNING: still counts, not converted to cumsums yet.
        int total_tail_counts = tail_exclusive_sums.at(NUM_THREADS, local_expert_idx);
        int stage_idx = local_expert_idx / NUM_EXPERT_PER_STAGE;
        reduce_inclusive_sums.at(cur_dp_idx, stage_idx) += total_tail_counts;
      }
    }
    __syncthreads();

    // ========Get the tail expert offset========
    // Exclusive sum of the last tail count row
    // little work. tid0 is enough.
    if (tid == 0) {
      // counts to exclusive cumsums.
      int sum = 0;
      int cur;
      for (int local_expert_idx = 0; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
        cur = tail_exclusive_sums.at(NUM_THREADS, local_expert_idx);
        tail_exclusive_sums.at(NUM_THREADS, local_expert_idx) = sum;
        sum += cur;
      }
      assert(sum == num_total_tokens_cur_dp);
    }
    __syncthreads();

    // ========Get Reduce Idx========
    // Counting Sort of experts with all tokens of this dp_idx
    TensorView<const int, 1> s_local_reduce_expert_bases_tensor(&tail_exclusive_sums.at(NUM_THREADS, 0), NUM_LOCAL_EXPERTS);

    for (int _recv_idx_offset = tid * num_recv_tokens_per_thread; _recv_idx_offset < recv_tokens_bound; ++_recv_idx_offset) {
      int recv_idx = dp_recv_idx_start + _recv_idx_offset;

      int tail_expert_idx = recv_idx_to_tail.at(recv_idx);
      const int EXPERT_BASE = s_local_reduce_expert_bases_tensor.at(tail_expert_idx);
      int& thread_expert_offset = tail_exclusive_sums.at(tid, tail_expert_idx);
      int dp_reduce_idx = EXPERT_BASE + thread_expert_offset;
      int reduce_idx = dp_reduce_idx + dp_recv_idx_start;

      // reduce to recved
      reduce_idx_to_recved_idx.at(reduce_idx) = recv_idx;
      // recved to reduce
      recv_idx_to_reduce_idx.at(recv_idx) = reduce_idx;

      thread_expert_offset++;
    }
    __syncthreads();
  }
  // ========Stage tail counts to inclusive cumsums========
  if (tid == 0) {
    for (int i = 1; i < WORLD_SIZE * NUM_STAGES; ++i) {
      p.reduce_plan.inclusive_sum[i] += p.reduce_plan.inclusive_sum[i - 1];
    }
  }
  __syncthreads();

  // ========Reduce Marker to Duplication Idx========
  // Reduce Marker: We have already have a reduce_idx(the mapping of reduce idx to recved idx).
  // For each token to reduce, their topk experts will be traversed.
  // If (reduce_idx, k_idx), related to (recved_idx, k_idx), routed to an EXPERT on this device, this (reduce_idx, k_idx) will be considered VALID.
  // We need a mapping of these valid reduce markers to the dup idx.
  // Duplication Idx: The order in the expert-sorted token copies.

  // Hence we have the mapping of dup_idx->recved_idx(many to one) but no recved_idx->dup_idx(one to many),
  // we are traversing dup_idx
  // dup_idx ->(expand for gemm plan)-> recved_marker -> recved_idx -> reduce_idx
  // NOTE: dup_idx < NUM_TOTAL_RECVED_TOKENS

  // default to INVALID VALUES
  for (int recved_idx = tid; recved_idx < NUM_TOTAL_RECVED_TOKENS; recved_idx += NUM_THREADS) {
    for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
      reduce_marker_to_dup_idx.at(recved_idx, k_idx) = NUM_TOTAL_RECVED_TOKENS * TOPK;
    }
  }
  __syncthreads();

  // traverse the duplicated tokens
  for (int dup_idx = tid; dup_idx < NUM_TOTAL_DUP_TOKENS; dup_idx += NUM_THREADS) {
    // dup_idx to recv_markers
    int recved_marker = p.expand_plan.mapping[dup_idx];
    int recved_idx = recved_marker / TOPK;
    int k_idx = recved_marker % TOPK;
    int reduce_idx = recv_idx_to_reduce_idx.at(recved_idx);
    reduce_marker_to_dup_idx.at(reduce_idx, k_idx) = dup_idx;
  }
}

__global__ void getLocalReducePlanKernel_ver2_decode(LocalReducePlanKernelParams p, LocalReducePlanKernelHelper helper) {
  // ========params========
  const int NUM_THREADS = 1;
  const int NUM_TOTAL_RECVED_TOKENS = p.inclusive_sum[p.ep_world_size * p.num_stages - 1];

  const int NUM_STAGES = p.num_stages;
  const int LOCAL_EXPERT_BEGIN = p.local_expert_begin;
  const int LOCAL_EXPERT_END = p.local_expert_end;
  const int NUM_LOCAL_EXPERTS = LOCAL_EXPERT_END - LOCAL_EXPERT_BEGIN;
  const int NUM_TOTAL_DUP_TOKENS = p.expand_plan.exclusive_sum[NUM_LOCAL_EXPERTS];
  assert(NUM_LOCAL_EXPERTS % NUM_STAGES == 0);
  const int NUM_EXPERT_PER_STAGE = NUM_LOCAL_EXPERTS / NUM_STAGES;
  const int WORLD_SIZE = p.ep_world_size;
  const int TOPK = p.topk;

  const int tid = threadIdx.x;

  // ========tensors========
  CREATE_EMPTY_STRUCT(__t0);
  TensorView<const int, 2, __t0> i_expert_indices(p.expert_indices, NUM_TOTAL_RECVED_TOKENS, TOPK);

  CREATE_EMPTY_STRUCT(__t1);
  TensorView<const int, 2, __t1> i_recv_inclusive_sums(p.inclusive_sum, WORLD_SIZE, NUM_STAGES);

  extern __shared__ int smem[];

  int* smem0 = smem;
  CREATE_EMPTY_STRUCT(__t2);
  // TensorView<int, 1, __t2> tmp_recv_idx_to_tail(p.tailing_experts, NUM_TOTAL_RECVED_TOKENS);
  TensorView<int, 1, __t2> s_recv_idx_to_tail(smem0, NUM_TOTAL_RECVED_TOKENS);

  int* smem1 = smem0 + LocalReducePlanKernelHelper::MAX_NUM_TOTAL_DUP_TOKENS;
  CREATE_EMPTY_STRUCT(__t3);
  // s_ means on shared memory
  // exc means exclusive cumsum
  TensorView<int, 2, __t3> s_tail_exclusive_sums(smem1, NUM_THREADS + 1, NUM_LOCAL_EXPERTS);

  int* smem2 = smem1 + LocalReducePlanKernelHelper::MAX_NUM_TOTAL_DUP_TOKENS;
  CREATE_EMPTY_STRUCT(__t4);
  // TensorView<int, 1, __t4> recv_idx_to_reduce_idx(p.sorted_tailing_experts, NUM_TOTAL_RECVED_TOKENS);
  TensorView<int, 1, __t4> tmp_recv_idx_to_reduce_idx(smem2, NUM_TOTAL_RECVED_TOKENS);

  // outputs
  CREATE_EMPTY_STRUCT(__t5);
  TensorView<int, 1, __t5> o_reduce_idx_to_recved_idx(p.reduce_plan.scale_mapping, NUM_TOTAL_RECVED_TOKENS);

  CREATE_EMPTY_STRUCT(__t6);
  TensorView<int, 2, __t6> o_reduce_marker_to_dup_idx(p.reduce_plan.mapping, NUM_TOTAL_RECVED_TOKENS, TOPK);

  CREATE_EMPTY_STRUCT(__t7);
  TensorView<int, 2, __t7> o_reduce_inclusive_sums(p.reduce_plan.inclusive_sum, WORLD_SIZE, NUM_STAGES);

  // NOTE: Reduce Idx indicates the pos of the token when choosing them to perform local reduction.
  // NOTE: Recved Idx indicates the pos of receiving the tokens, related to Leading Experts and the Dispatch Comminication Process

  // ========Process each DP rank, from where the token comes========
  // There are two outcomes that should be processed for each single rank
  // a. DP-stage tailing expert cumcums.
  // At each stage of EPS, we process tokens for several different ranks.
  // Thus, the tailing expert cumsums has 2 dims, recved_from_dp_idx and stage_idx.
  // Meaning that we need to process some tokens for recved_from_dp_idx at stage_idx.
  // We first count the tails of each RANK and STAGE in shared memory, then perform inclusive cumsum of all ranks after the tokens recved from each rank.
  // b. Inner DP Reduce idx to Recved Idx

  // for each DP
  for (int cur_dp_idx = 0; cur_dp_idx < WORLD_SIZE; ++cur_dp_idx) {
    // ========Init========
    // tailing experts
    constexpr int INVALID_TAIL = INT_MAX;
    for (int recv_idx = tid; recv_idx < NUM_TOTAL_RECVED_TOKENS; recv_idx += NUM_THREADS) {
      s_recv_idx_to_tail.at(recv_idx) = INVALID_TAIL;
    }
    // __syncthreads();

    const int dp_recv_idx_start = cur_dp_idx == 0 ? 0 : i_recv_inclusive_sums.at(cur_dp_idx - 1, NUM_STAGES - 1);
    const int dp_recv_idx_next_start = i_recv_inclusive_sums.at(cur_dp_idx, NUM_STAGES - 1);
    // WARNING: Do not plus one when using cumsum!
    const int num_total_tokens_cur_dp = dp_recv_idx_next_start - dp_recv_idx_start;
    // traverse all the recved tokens from this dp idx
    // each thread work on several tokens

    // ========Get tailing expert on this device========
    const int num_recv_tokens_per_thread = CEILDIV(num_total_tokens_cur_dp, NUM_THREADS);
    const int recv_tokens_bound = std::min(num_total_tokens_cur_dp, (tid + 1) * num_recv_tokens_per_thread);
    // Each THREAD checks its own portion of tokens
    for (int _recv_idx_offset = tid * num_recv_tokens_per_thread; _recv_idx_offset < recv_tokens_bound; ++_recv_idx_offset) {
      // get the idx by Base + Offset
      int recv_idx = dp_recv_idx_start + _recv_idx_offset;

      for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
        int _expert_id = i_expert_indices.at(recv_idx, k_idx);
        if (LOCAL_EXPERT_BEGIN <= _expert_id && _expert_id < LOCAL_EXPERT_END) {
          int local_expert_idx = _expert_id - LOCAL_EXPERT_BEGIN;
          int& cur_tail_idx = s_recv_idx_to_tail.at(recv_idx);
          if (cur_tail_idx == INVALID_TAIL || cur_tail_idx < local_expert_idx) {
            cur_tail_idx = local_expert_idx;
          }
        }
      }
    }
    // __syncthreads();
    // ========Count tail on each expert========
    // memset
    for (int local_expert_idx = 0; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
      s_tail_exclusive_sums.at(tid + 1, local_expert_idx) = 0;
      if (tid == 0) {
        s_tail_exclusive_sums.at(0, local_expert_idx) = 0;
      }
    }
    // __syncthreads();

    for (int _recv_idx_offset = tid * num_recv_tokens_per_thread; _recv_idx_offset < recv_tokens_bound; ++_recv_idx_offset) {
      // get the idx by Base + Offset
      int recv_idx = dp_recv_idx_start + _recv_idx_offset;
      int tail_expert_idx = s_recv_idx_to_tail.at(recv_idx);
      // NOTE: we write to row of tid+1, for we need the exclusive sum of each expert. It can be used to calc the base start of each EXPERT for each THREAD.
      // WARNING: We put the counts first. Then they will be transformed into cumsums
      if (tail_expert_idx != INVALID_TAIL) {
        s_tail_exclusive_sums.at(tid + 1, tail_expert_idx) += 1;
      }
    }
    // __syncthreads();

    // ========Exclusive Cumsum of rows========
    // NOTE: tid's role temporarily changes. Each THREAD works on several columns to ensure parallelism.
    for (int local_expert_idx = tid; local_expert_idx < NUM_LOCAL_EXPERTS; local_expert_idx += NUM_THREADS) {
      for (int row_idx = 1; row_idx <= NUM_THREADS; ++row_idx) {
        s_tail_exclusive_sums.at(row_idx, local_expert_idx) += s_tail_exclusive_sums.at(row_idx - 1, local_expert_idx);
      }
    }
    // __syncthreads();

    // ========Write stage tail counts to output========
    // little work. t0 is enough.
    if (tid == 0) {
      // memset
      for (int stage_idx = 0; stage_idx < NUM_STAGES; ++stage_idx) {
        o_reduce_inclusive_sums.at(cur_dp_idx, stage_idx) = 0;
      }

      for (int local_expert_idx = 0; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
        // WARNING: still counts, not converted to cumsums yet.
        int total_tail_counts = s_tail_exclusive_sums.at(NUM_THREADS, local_expert_idx);
        int stage_idx = local_expert_idx / NUM_EXPERT_PER_STAGE;
        o_reduce_inclusive_sums.at(cur_dp_idx, stage_idx) += total_tail_counts;
      }
    }
    // __syncthreads();

    // ========Get the tail expert offset========
    // Exclusive sum of the last tail count row
    // little work. tid0 is enough.
    if (tid == 0) {
      // counts to exclusive cumsums.
      int sum = 0;
      int cur;
      for (int local_expert_idx = 0; local_expert_idx < NUM_LOCAL_EXPERTS; ++local_expert_idx) {
        cur = s_tail_exclusive_sums.at(NUM_THREADS, local_expert_idx);
        s_tail_exclusive_sums.at(NUM_THREADS, local_expert_idx) = sum;
        sum += cur;
      }
      assert(sum == num_total_tokens_cur_dp);
    }
    // __syncthreads();

    // ========Get Reduce Idx========
    // Counting Sort of experts with all tokens of this dp_idx
    TensorView<const int, 1> s_local_reduce_expert_bases_tensor(&s_tail_exclusive_sums.at(NUM_THREADS, 0), NUM_LOCAL_EXPERTS);

    for (int _recv_idx_offset = tid * num_recv_tokens_per_thread; _recv_idx_offset < recv_tokens_bound; ++_recv_idx_offset) {
      int recv_idx = dp_recv_idx_start + _recv_idx_offset;

      int tail_expert_idx = s_recv_idx_to_tail.at(recv_idx);
      const int EXPERT_BASE = s_local_reduce_expert_bases_tensor.at(tail_expert_idx);
      int& thread_expert_offset = s_tail_exclusive_sums.at(tid, tail_expert_idx);
      int dp_reduce_idx = EXPERT_BASE + thread_expert_offset;
      int reduce_idx = dp_reduce_idx + dp_recv_idx_start;

      // reduce to recved
      o_reduce_idx_to_recved_idx.at(reduce_idx) = recv_idx;
      // recved to reduce
      tmp_recv_idx_to_reduce_idx.at(recv_idx) = reduce_idx;

      thread_expert_offset++;
    }
    // __syncthreads();
  }
  // ========Stage tail counts to inclusive cumsums========
  if (tid == 0) {
    for (int i = 1; i < WORLD_SIZE * NUM_STAGES; ++i) {
      p.reduce_plan.inclusive_sum[i] += p.reduce_plan.inclusive_sum[i - 1];
    }
  }
  // __syncthreads();

  // ========Reduce Marker to Duplication Idx========
  // Reduce Marker: We have already have a reduce_idx(the mapping of reduce idx to recved idx).
  // For each token to reduce, their topk experts will be traversed.
  // If (reduce_idx, k_idx), related to (recved_idx, k_idx), routed to an EXPERT on this device, this (reduce_idx, k_idx) will be considered VALID.
  // We need a mapping of these valid reduce markers to the dup idx.
  // Duplication Idx: The order in the expert-sorted token copies.

  // Hence we have the mapping of dup_idx->recved_idx(many to one) but no recved_idx->dup_idx(one to many),
  // we are traversing dup_idx
  // dup_idx ->(expand for gemm plan)-> recved_marker -> recved_idx -> reduce_idx
  // NOTE: dup_idx < NUM_TOTAL_RECVED_TOKENS

  // default to INVALID VALUES
  for (int recved_idx = tid; recved_idx < NUM_TOTAL_RECVED_TOKENS; recved_idx += NUM_THREADS) {
    for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
      o_reduce_marker_to_dup_idx.at(recved_idx, k_idx) = NUM_TOTAL_RECVED_TOKENS * TOPK;
    }
  }
  // __syncthreads();

  // traverse the duplicated tokens
  for (int dup_idx = tid; dup_idx < NUM_TOTAL_DUP_TOKENS; dup_idx += NUM_THREADS) {
    // dup_idx to recv_markers
    int recved_marker = p.expand_plan.mapping[dup_idx];
    int recved_idx = recved_marker / TOPK;
    int k_idx = recved_marker % TOPK;
    int reduce_idx = tmp_recv_idx_to_reduce_idx.at(recved_idx);
    o_reduce_marker_to_dup_idx.at(reduce_idx, k_idx) = dup_idx;
  }
}

void getLocalReducePlan_ver2(LocalReducePlanKernelParams p, cudaStream_t stream, int num_global_tokens) {
  // num_global_tokens is the runtime exect number
  LocalReducePlanKernelHelper helper(p);
  const size_t max_num_routed_tokens = num_global_tokens * p.topk;
  if (max_num_routed_tokens > LocalReducePlanKernelHelper::MAX_NUM_TOTAL_DUP_TOKENS) {
    const size_t dyn_smem_bytes = helper.s_threads_tail_counts_numel(helper.num_threads_) * sizeof(int);
    cudaFuncSetAttribute(getLocalReducePlanKernel_ver2, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem_bytes);
    getLocalReducePlanKernel_ver2<<<1, helper.num_threads_, dyn_smem_bytes, stream>>>(p, helper);
  } else {
    constexpr size_t num_threads_fast = 1;
    const size_t smem_bytes = LocalReducePlanKernelHelper::MAX_NUM_TOTAL_DUP_TOKENS * sizeof(int) * 3;
    getLocalReducePlanKernel_ver2_decode<<<1, num_threads_fast, smem_bytes, stream>>>(p, helper);
  }
}

}  // namespace eps
