#include "src/scheduler/local_reducer.cuh"
#include "src/scheduler/common.cuh"

#include <cub/cub.cuh>

#include "src/common/cuda_utils.cuh"
#include "src/scheduler/utils.cuh"

namespace eps {

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void getLocalReducePlanKernel(LocalReducePlanKernelParams p) {
  int num_tokens = p.inclusive_sum[p.ep_world_size * p.num_stages - 1];
  assert(BLOCK_SIZE == blockDim.x);
  assert(BLOCK_SIZE * ITEMS_PER_THREAD >= num_tokens);

  typedef cub::BlockLoad<int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT> BlockLoad;
  typedef cub::BlockRadixSort<int, BLOCK_SIZE, ITEMS_PER_THREAD, int> BlockRadixSort;
  typedef cub::BlockStore<int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_DIRECT> BlockStore;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
    typename BlockRadixSort::TempStorage sort;
  } temp_storage;

  //// Extract tailing experts
  for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
    int tailing_expert = -1;
    for (int j = 0; j < p.topk; j++) {
      int expert = p.expert_indices[i * p.topk + j];
      if (p.local_expert_begin <= expert && expert < p.local_expert_end && expert > tailing_expert) {
        tailing_expert = expert;
      }
    }
    p.tailing_experts[i] = tailing_expert;
  }
  __syncthreads();

  //// by device 分段 sorting
  for (int device = 0; device < p.ep_world_size; device++) {
    int begin = device == 0 ? 0 : p.inclusive_sum[device * p.num_stages - 1];
    int end = p.inclusive_sum[device * p.num_stages + p.num_stages - 1];
    int num_tokens_device = end - begin;
    if (num_tokens_device > 0) {
      int thread_tailing_experts[ITEMS_PER_THREAD];
      BlockLoad(temp_storage.load)
          .Load(p.tailing_experts + begin, thread_tailing_experts, num_tokens_device, p.num_experts);
      __syncthreads(); // Barrier for smem reuse

      int thread_values[ITEMS_PER_THREAD];
#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_values[i] = begin + std::min(int(ITEMS_PER_THREAD * threadIdx.x + i), num_tokens_device);
      }

      BlockRadixSort(temp_storage.sort).Sort(thread_tailing_experts, thread_values);
      __syncthreads(); // Barrier for smem reuse

      BlockStore(temp_storage.store).Store(p.sorted_tailing_experts + begin, thread_tailing_experts, num_tokens_device);
      __syncthreads(); // Barrier for smem reuse

      BlockStore(temp_storage.store).Store(p.reduce_plan.scale_mapping + begin, thread_values, num_tokens_device);
      __syncthreads(); // Barrier for smem reuse

      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (thread_values[i] < begin + num_tokens_device) {
          p.mapping[thread_values[i]] = begin + threadIdx.x * ITEMS_PER_THREAD + i;
        }
      }

      if (threadIdx.x < p.num_stages) {
        int num_experts_per_stage = (p.local_expert_end - p.local_expert_begin) / p.num_stages;
        int target = p.local_expert_begin + threadIdx.x * num_experts_per_stage + num_experts_per_stage - 1;
        p.reduce_plan.inclusive_sum[device * p.num_stages + threadIdx.x] =
            begin + find_total_elts_leq_target(p.sorted_tailing_experts + begin, num_tokens_device, target);
      }
    } else {
      if (threadIdx.x < p.num_stages) {
        p.reduce_plan.inclusive_sum[device * p.num_stages + threadIdx.x] = begin;
      }
    }
  }
  __syncthreads();

  //// Reverse Mapping
  for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
    for (int j = 0; j < p.topk; j++) {
      p.reduce_plan.mapping[i * p.topk + j] = num_tokens * p.topk;
    }
  }
  __syncthreads();

  int num_expanded = p.expand_plan.exclusive_sum[p.local_expert_end - p.local_expert_begin];
  for (int i = threadIdx.x; i < num_expanded; i += blockDim.x) {
    int token_idx = p.expand_plan.mapping[i] / p.topk;
    int topk_idx = p.expand_plan.mapping[i] % p.topk;
    p.reduce_plan.mapping[p.mapping[token_idx] * p.topk + topk_idx] = i;
  }
}

template <typename T, typename U, int BLOCK_SIZE, int HIDDEN_SIZE>
__global__ void localReduceKernel(LocalReduceKernelParams<T, U> p, int stage_idx)
{
  assert(gridDim.x % p.ep_world_size == 0);
  assert(BLOCK_SIZE == blockDim.x);

  extern __shared__ char smem[]; /*topk*/
  U* scales = reinterpret_cast<U*>(smem);

  int num_blocks_per_device = gridDim.x / p.ep_world_size;
  int device_idx = blockIdx.x / num_blocks_per_device;

  int begin = (device_idx == 0 && stage_idx == 0)
                  ? 0
                  : p.reduce_plan.inclusive_sum[device_idx * p.num_stages + stage_idx - 1];
  int end = p.reduce_plan.inclusive_sum[device_idx * p.num_stages + stage_idx];

  using VectorizedType = VectorizedType<T>;
  constexpr int ItemsPerVec = VectorizedType::ItemsPerVec;
  constexpr int VecsPerThread = CEILDIV(HIDDEN_SIZE, BLOCK_SIZE * ItemsPerVec);

  static_assert(HIDDEN_SIZE % ItemsPerVec == 0);

  VectorizedType thread_result[VecsPerThread];
  auto clear = [&thread_result]()
  {
    #pragma unroll
    for (int i = 0; i < VecsPerThread; i++)
    {
      thread_result[i] = VectorizedType{};
    }
  };

  auto reduce = [&thread_result](VectorizedType thread_value, int i, U scale)
  {
    thread_result[i] += (thread_value * scale);
  };

  int num_tokens = p.reduce_plan.inclusive_sum[p.ep_world_size * p.num_stages - 1];
  for (int i = blockIdx.x % num_blocks_per_device; i < end - begin; i += num_blocks_per_device)
  {
    __syncthreads();
    if (threadIdx.x == 0)
    {
      for (int j = 0; j < p.topk; j++)
      {
        int expanded_row_idx = p.reduce_plan.mapping[(begin + i) * p.topk + j];
        if (expanded_row_idx < num_tokens * p.topk)
        {
          U scale = p.expert_scales[p.reduce_plan.scale_mapping[begin + i] * p.topk + j];
          scales[j] = scale;
        }
      }
    }
    __syncthreads();

    clear();
    for (int j = 0; j < p.topk; j++)
    {
      int64_t expanded_row_idx = p.reduce_plan.mapping[(begin + i) * p.topk + j];
      if (expanded_row_idx < num_tokens * p.topk)
      {
        U scale = scales[j];
        const T *src = p.src + p.hidden_size * expanded_row_idx;
        #pragma unroll
        for (int k = threadIdx.x; k < HIDDEN_SIZE / ItemsPerVec; k += BLOCK_SIZE)
        {
          VectorizedType thread_value = ((VectorizedType *)src)[k];
          reduce(thread_value, k / BLOCK_SIZE, scale);
        }
      }
    }

    T *dst = p.dst + p.hidden_size * (begin + i);
    #pragma unroll
    for (int k = threadIdx.x; k < HIDDEN_SIZE / ItemsPerVec; k += BLOCK_SIZE)
    {
      ((VectorizedType *)dst)[k] = thread_result[k / BLOCK_SIZE];
    }
  }
}

void getLocalReducePlan(LocalReducePlanKernelParams p, cudaStream_t stream) {
  static constexpr int BLOCK_SIZE = 1024;
  static constexpr int ITEMS_PER_THREAD = 4;

  getLocalReducePlanKernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<1, BLOCK_SIZE, 0, stream>>>(p);
}

template <typename T, typename U>
void localReduce(LocalReduceKernelParams<T, U> p, int stage_idx, cudaStream_t stream) {

  int num_blocks;
  int block_size;
  auto num_blocks_heuristic = [&](int64_t num_tokens_hint, int ep_world_size, int num_stages)
  {
    num_blocks = (std::max(num_tokens_hint / num_stages, 1L) + ep_world_size - 1) / ep_world_size * ep_world_size;
    block_size = 512;
    if (num_blocks > 256)
    {
      block_size = 256;
    }
  };
  num_blocks_heuristic(p.num_tokens_hint, p.ep_world_size, p.num_stages);

  BLOCK_SIZE_SWITCH(block_size,
    HIDDEN_SIZE_SWITCH(p.hidden_size,
      [&]() {
        localReduceKernel<T, U, BLOCK_SIZE, HIDDEN_SIZE><<<num_blocks, BLOCK_SIZE, sizeof(U) * p.topk, stream>>>(p, stage_idx);
      })
  )();
}

template
void localReduce<float, float>(LocalReduceKernelParams<float, float> p, int stage_idx, cudaStream_t stream);

template
void localReduce<half, half>(LocalReduceKernelParams<half, half> p, int stage_idx, cudaStream_t stream);
template
void localReduce<half, float>(LocalReduceKernelParams<half, float> p, int stage_idx, cudaStream_t stream);

#ifdef ENABLE_BF16
template
void localReduce<__nv_bfloat16, __nv_bfloat16>(LocalReduceKernelParams<__nv_bfloat16, __nv_bfloat16> p, int stage_idx, cudaStream_t stream);
template
void localReduce<__nv_bfloat16, float>(LocalReduceKernelParams<__nv_bfloat16, float> p, int stage_idx, cudaStream_t stream);
#endif

} // namespace eps
