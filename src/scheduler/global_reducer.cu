#include "src/scheduler/global_reducer.cuh"
#include "src/scheduler/common.cuh"

#include <cub/cub.cuh>


#include "src/common/cuda_utils.cuh"
#include "src/scheduler/utils.cuh"

namespace eps {

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void getGlobalReducePlanKernel(GlobalReducePlanKernelParams p)
{
    assert(BLOCK_SIZE == blockDim.x);
    assert(BLOCK_SIZE * ITEMS_PER_THREAD >= p.num_tokens);

    typedef cub::BlockLoad<int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT> BlockLoad;
    typedef cub::BlockRadixSort<int, BLOCK_SIZE, ITEMS_PER_THREAD, int> BlockRadixSort;
    typedef cub::BlockStore<int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_DIRECT> BlockStore;

    __shared__ union
    {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
        typename BlockRadixSort::TempStorage sort;
    } temp_storage;

    //// 分段 Extract tailing experts
    for (int device = 0; device < p.ep_world_size; device++)
    {
        int begin = device == 0 ? 0 : p.expand_plan.inclusive_sum[device * p.num_stages - 1];
        int end = p.expand_plan.inclusive_sum[device * p.num_stages + p.num_stages - 1];
        int local_expert_begin = device * p.num_experts / p.ep_world_size;
        int local_expert_end = (device + 1) * p.num_experts / p.ep_world_size;

        for (int i = threadIdx.x; i < end - begin; i += blockDim.x)
        {
            int token_idx = p.expand_plan.mapping[begin + i] % p.num_tokens;

            int tailing_expert = -1;
            for (int j = 0; j < p.topk; j++)
            {
                int expert = p.expert_indices[token_idx * p.topk + j];
                if (local_expert_begin <= expert && expert < local_expert_end && expert > tailing_expert)
                {
                    tailing_expert = expert;
                }
            }
            p.tailing_experts[begin + i] = tailing_expert;
        }
    }
    __syncthreads();

    //// Reset mapping
    for (int i = threadIdx.x; i < p.num_tokens; i += blockDim.x)
    {
        for (int j = 0; j < p.ep_world_size; j++)
        {
            p.reduce_plan.mapping[i * p.ep_world_size + j] = p.num_tokens * p.ep_world_size;
        }
    }
    __syncthreads();

    //// by device 分段 sorting
    for (int device = 0; device < p.ep_world_size; device++)
    {
        int begin = device == 0 ? 0 : p.expand_plan.inclusive_sum[device * p.num_stages - 1];
        int end = p.expand_plan.inclusive_sum[device * p.num_stages + p.num_stages - 1];
        int local_expert_begin = device * p.num_experts / p.ep_world_size;
        int local_expert_end = (device + 1) * p.num_experts / p.ep_world_size;
        int num_expanded = end - begin;

        if (num_expanded > 0)
        {
            int thread_tailing_experts[ITEMS_PER_THREAD];
            BlockLoad(temp_storage.load).Load(p.tailing_experts + begin, thread_tailing_experts, num_expanded, p.num_experts);
            __syncthreads(); // Barrier for smem reuse

            int thread_expand_plan[ITEMS_PER_THREAD];
            BlockLoad(temp_storage.load).Load(p.expand_plan.mapping + begin, thread_expand_plan, num_expanded, p.num_tokens * p.ep_world_size);
            __syncthreads(); // Barrier for smem reuse

            BlockRadixSort(temp_storage.sort).Sort(thread_tailing_experts, thread_expand_plan);
            __syncthreads(); // Barrier for smem reuse

            BlockStore(temp_storage.store).Store(p.sorted_tailing_experts + begin, thread_tailing_experts, num_expanded);
            __syncthreads(); // Barrier for smem reuse

            if (threadIdx.x < p.num_stages)
            {
                int num_experts_per_stage = (local_expert_end - local_expert_begin) / p.num_stages;
                int target = local_expert_begin + threadIdx.x * num_experts_per_stage + num_experts_per_stage - 1;
                p.reduce_plan.inclusive_sum[device * p.num_stages + threadIdx.x] = begin + find_total_elts_leq_target(p.sorted_tailing_experts + begin, num_expanded, target);
            }

            for (int i = 0; i < ITEMS_PER_THREAD; i++)
            {
                if (thread_expand_plan[i] < p.num_tokens * p.ep_world_size)
                {
                    int token_idx = thread_expand_plan[i] % p.num_tokens;
                    int device_idx = thread_expand_plan[i] / p.num_tokens;
                    assert(device == device_idx);
                    p.reduce_plan.mapping[token_idx * p.ep_world_size + device_idx] = begin + threadIdx.x * ITEMS_PER_THREAD + i;
                }
            }
        }
        else
        {
            if (threadIdx.x < p.num_stages)
            {
                p.reduce_plan.inclusive_sum[device * p.num_stages + threadIdx.x] = begin;
            }
        }
    }
}

void getGlobalReducePlan(GlobalReducePlanKernelParams p, cudaStream_t stream) {
  static constexpr int BLOCK_SIZE = 1024;
  static constexpr int ITEMS_PER_THREAD = 4;

  getGlobalReducePlanKernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<1, BLOCK_SIZE, 0, stream>>>(p);
}

template <typename T, int BLOCK_SIZE, int HIDDEN_SIZE>
__global__ void globalReduceKernel(GlobalReduceKernelParams<T> p) {
  assert(BLOCK_SIZE == blockDim.x);

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

  auto reduce = [&thread_result](VectorizedType thread_value, int i)
  {
    thread_result[i] += thread_value;
  };

  for (int token_idx = blockIdx.x; token_idx < p.num_tokens; token_idx += gridDim.x)
  {
    clear();

    for (int device_idx = 0; device_idx < p.ep_world_size; device_idx++)
    {
      int expanded_row_idx = p.reduce_plan.mapping[token_idx * p.ep_world_size + device_idx];
      if (expanded_row_idx < p.num_tokens * p.ep_world_size)
      {
        const T *src = p.src + p.hidden_size * expanded_row_idx;

        #pragma unroll
        for (int k = threadIdx.x; k < HIDDEN_SIZE / ItemsPerVec; k += BLOCK_SIZE)
        {
          VectorizedType thread_value = ((VectorizedType *)src)[k];
          reduce(thread_value, k / BLOCK_SIZE);
        }
      }
    }

    T *dst = p.dst + p.hidden_size * token_idx;
    const T* shared = p.shared + p.hidden_size * token_idx;
    #pragma unroll
    for (int k = threadIdx.x; k < HIDDEN_SIZE / ItemsPerVec; k += BLOCK_SIZE)
    {
      if (p.shared == nullptr) {
        ((VectorizedType *)dst)[k] = thread_result[k / BLOCK_SIZE];
      } else {
        ((VectorizedType *)dst)[k] = ((VectorizedType *)shared)[k] + thread_result[k / BLOCK_SIZE];
      }
    }
  }
}

template<typename T>
void globalReduce(GlobalReduceKernelParams<T> p, cudaStream_t stream) {
  if (p.num_tokens > 0) {
    int num_blocks;
    int block_size;
    auto num_blocks_heuristic = [&](int64_t num_tokens)
    {
      num_blocks = num_tokens;
      block_size = 1024;
      if (num_blocks > 256)
      {
        block_size = 512;
      }
      num_blocks = std::min(num_blocks, 1024);
    };

    num_blocks_heuristic(p.num_tokens);

    BLOCK_SIZE_SWITCH(block_size,
      HIDDEN_SIZE_SWITCH(p.hidden_size,
        [&]() {
          globalReduceKernel<T, BLOCK_SIZE, HIDDEN_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(p);
        })
    )();
  }
}

template
void globalReduce(GlobalReduceKernelParams<float> p, cudaStream_t stream);

template
void globalReduce(GlobalReduceKernelParams<half> p, cudaStream_t stream);

#ifdef ENABLE_BF16
template
void globalReduce(GlobalReduceKernelParams<__nv_bfloat16> p, cudaStream_t stream);
#endif

}
