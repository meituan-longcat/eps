#include "src/scheduler/comm_expander.cuh"
#include "src/scheduler/common.cuh"

#include <cub/cub.cuh>

#include "src/common/cuda_utils.cuh"
#include "src/scheduler/utils.cuh"

namespace eps {

__device__ int waitPreviousWriteCurrent(unsigned long long int *tile_states, int bid, int value, int num_blocks)
{
    __shared__ int previous_sum;
    if (threadIdx.x == 0)
    {
        unsigned long long int prev_state{};
        if (bid > 0)
        {
            do
            {
                prev_state = atomicAdd(&tile_states[bid], 0);
            } while (prev_state == 0);
        }

        previous_sum = prev_state & 0xFFFFFFFF;

        unsigned long long int current_state = (1ull << 32) | (previous_sum + value);

        if (bid < num_blocks - 1)
        {
            atomicAdd(&tile_states[bid + 1], current_state);
        }
    }

    __syncthreads();
    return previous_sum;
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void getExpandForCommPlanKernel(ExpandForCommPlanKernelParams p) {
  int ep_world_size = gridDim.x;
  assert(BLOCK_SIZE == blockDim.x);
  assert(BLOCK_SIZE * ITEMS_PER_THREAD >= p.num_tokens);
  // assert(32 >= p.num_stages);

  typedef cub::BlockRadixSort<int, BLOCK_SIZE, ITEMS_PER_THREAD, int> BlockRadixSort;
  typedef cub::BlockStore<int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE> BlockStore;
  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;

  __shared__ union {
    typename BlockStore::TempStorage store;
    typename BlockRadixSort::TempStorage sort;
    typename BlockReduce::TempStorage reduce;
  } temp_storage;
  __shared__ int block_valid;

  int expert_begin = (p.num_experts / p.ep_world_size) * blockIdx.x;
  int expert_end = (p.num_experts / p.ep_world_size) * (blockIdx.x + 1);

  int thread_leading_experts[ITEMS_PER_THREAD];
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    int token_idx = threadIdx.x * ITEMS_PER_THREAD + i;
    int leading_expert = p.num_experts;
    if (token_idx < p.num_tokens)
    {
      for (int j = 0; j < p.topk; j++)
      {
        int expert = p.expert_indices[token_idx * p.topk + j];
        if (expert_begin <= expert && expert < expert_end && expert < leading_expert)
        {
          leading_expert = expert;
        }
      }
    }

    thread_leading_experts[i] = leading_expert;
  }

  int thread_valid = 0;
  int thread_expand_plan[ITEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_expand_plan[i] =
        std::min(int(blockIdx.x * p.num_tokens + ITEMS_PER_THREAD * threadIdx.x + i), p.ep_world_size * p.num_tokens);
    thread_valid += (thread_leading_experts[i] < p.num_experts);
  }

  // Compute the block-wide sum for thread0
  int block_valid_ = BlockReduce(temp_storage.reduce).Sum(thread_valid);
  if (threadIdx.x == 0) {
    block_valid = block_valid_;
  }
  __syncthreads(); // Barrier for smem reuse

  BlockRadixSort(temp_storage.sort).Sort(thread_leading_experts, thread_expand_plan);
  __syncthreads(); // Barrier for smem reuse

  int exclusive_prefix_sum = waitPreviousWriteCurrent(p.tile_states, blockIdx.x, block_valid, ep_world_size);

  BlockStore(temp_storage.store)
      .Store(p.sorted_leading_experts + exclusive_prefix_sum, thread_leading_experts, block_valid);
  __syncthreads(); // Barrier for smem reuse

  BlockStore(temp_storage.store).Store(p.expand_plan.mapping + exclusive_prefix_sum, thread_expand_plan, block_valid);
  __syncthreads(); // Barrier for smem reuse

  if (threadIdx.x < p.num_stages) {
    int target = (p.num_experts / ep_world_size) * blockIdx.x +
        (threadIdx.x + 1) * (p.num_experts / ep_world_size / p.num_stages) - 1;
    int prefix_sum = exclusive_prefix_sum +
        find_total_elts_leq_target(p.sorted_leading_experts + exclusive_prefix_sum, block_valid, target);
    p.expand_plan.inclusive_sum[blockIdx.x * p.num_stages + threadIdx.x] = prefix_sum;
  }
}

template <typename T, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void expandForCommKernel(ExpandForCommKernelParams<T> p) {
  assert(BLOCK_SIZE == blockDim.x);

  typedef cub::BlockLoad<T, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
  typedef cub::BlockStore<T, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE> BlockStore;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
  } temp_storage;
  int num_expanded = p.expand_plan.inclusive_sum[p.ep_world_size * p.num_stages - 1];

  T thread_value[ITEMS_PER_THREAD];
  for (int i = blockIdx.x; i < num_expanded; i += gridDim.x) {
    const T* src = p.src + p.cols * (p.expand_plan.mapping[i] % p.num_tokens);
    T* dst = p.dst + p.cols * i;
    for (int k = 0; k < p.cols; k += ITEMS_PER_THREAD * blockDim.x)
    {
      BlockLoad(temp_storage.load).Load(src + k, thread_value, p.cols - k);
      __syncthreads(); // Barrier for smem reuse

      BlockStore(temp_storage.store).Store(dst + k, thread_value, p.cols - k);
      __syncthreads(); // Barrier for smem reuse
    }
  }
}

template <typename T, int BLOCK_SIZE, int HIDDEN_SIZE>
__global__ void expandForCommKernelVectorized(ExpandForCommKernelParams<T> p) {
  assert(BLOCK_SIZE == blockDim.x);

  int num_expanded = p.expand_plan.inclusive_sum[p.ep_world_size * p.num_stages - 1];

  using VectorizedType = VectorizedType<T>;
  constexpr int ItemsPerVec = VectorizedType::ItemsPerVec;
  static_assert(HIDDEN_SIZE % ItemsPerVec == 0);

  for (int i = blockIdx.x; i < num_expanded; i += gridDim.x) {
    const T* src = p.src + p.cols * (p.expand_plan.mapping[i] % p.num_tokens);
    T* dst = p.dst + p.cols * i;

    #pragma unroll
    for (int k = threadIdx.x; k < HIDDEN_SIZE / ItemsPerVec; k += BLOCK_SIZE)
    {
      reinterpret_cast<VectorizedType *>(dst)[k] = reinterpret_cast<const VectorizedType *>(src)[k];
    }
  }
}

void getExpandForCommPlan(ExpandForCommPlanKernelParams p, cudaStream_t stream) {
  static constexpr int BLOCK_SIZE = 1024;
  static constexpr int ITEMS_PER_THREAD = 4;

  getExpandForCommPlanKernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<p.ep_world_size, BLOCK_SIZE, 0, stream>>>(p);
}

template <typename T>
void expandForComm(ExpandForCommKernelParams<T> p, cudaStream_t stream) {
  if (p.num_tokens > 0) {
    int num_blocks;
    int block_size;
    auto num_blocks_heuristic = [&](int64_t num_tokens, int topk, int ep_world_size, int cols)
    {
      num_blocks = std::max(num_tokens * std::min(topk, ep_world_size), 1L);
      if (cols >= 1024)
      {
        block_size = 512;
        if (num_blocks > 256)
        {
          block_size = 256;
        }
        num_blocks = std::min(num_blocks, 1024);
      }
      else
      {
        block_size = 128;
        num_blocks = std::min(num_blocks, 2048);
      }
    };
    num_blocks_heuristic(p.num_tokens, p.topk, p.ep_world_size, p.cols);

    if (p.cols >= 1024) {
      BLOCK_SIZE_SWITCH(block_size,
        HIDDEN_SIZE_SWITCH(p.cols,
          [&]() {
            expandForCommKernelVectorized<T, BLOCK_SIZE, HIDDEN_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(p);
          })
      )();
    } else {
      BLOCK_SIZE_SWITCH(block_size,
        [&]() {
          static constexpr int ITEMS_PER_THREAD = 8;
          expandForCommKernel<T, BLOCK_SIZE, ITEMS_PER_THREAD><<<num_blocks, BLOCK_SIZE, 0, stream>>>(p);
        }
      )();
    }
  }
}

template
void expandForComm(ExpandForCommKernelParams<float> p, cudaStream_t stream);

template
void expandForComm(ExpandForCommKernelParams<half> p, cudaStream_t stream);

#ifdef ENABLE_BF16
template
void expandForComm(ExpandForCommKernelParams<__nv_bfloat16> p, cudaStream_t stream);
#endif

template
void expandForComm(ExpandForCommKernelParams<__nv_fp8_e4m3> p, cudaStream_t stream);

template
void expandForComm(ExpandForCommKernelParams<int> p, cudaStream_t stream);

} // namespace eps