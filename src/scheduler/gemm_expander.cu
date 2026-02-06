#include "src/scheduler/gemm_expander.cuh"
#include "src/scheduler/common.cuh"

#include <cub/cub.cuh>

#include "src/common/cuda_utils.cuh"
#include "src/scheduler/utils.cuh"

namespace eps {

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void getExpandForGemmPlanKernel(ExpandForGemmPlanKernelParams p) {
  int num_tokens = p.inclusive_sum[p.ep_world_size * p.num_stages - 1];
  assert(BLOCK_SIZE == blockDim.x);
  // if (threadIdx.x == 0) {
  //   printf("num_tokens: %d, p.topk: %d\n", num_tokens, p.topk);
  // }
  assert(BLOCK_SIZE * ITEMS_PER_THREAD >= num_tokens * p.topk);

  typedef cub::BlockLoad<int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT> BlockLoad;
  typedef cub::BlockRadixSort<int, BLOCK_SIZE, ITEMS_PER_THREAD, int> BlockRadixSort;
  typedef cub::BlockStore<int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_DIRECT> BlockStore;
  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
    typename BlockRadixSort::TempStorage sort;
    typename BlockReduce::TempStorage reduce;
  } temp_storage;
  __shared__ int block_valid;

  int thread_experts[ITEMS_PER_THREAD];
  BlockLoad(temp_storage.load).Load(p.expert_indices, thread_experts, num_tokens * p.topk, p.num_experts);
  __syncthreads(); // Barrier for smem reuse

  int thread_valid = 0;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (thread_experts[i] < p.local_expert_begin || thread_experts[i] >= p.local_expert_end) {
      thread_experts[i] = p.num_experts;
    } else {
      thread_valid++;
    }
  }

  // Compute the block-wide sum for thread0
  int block_valid_ = BlockReduce(temp_storage.reduce).Sum(thread_valid);
  if (threadIdx.x == 0) {
    block_valid = block_valid_;
  }
  __syncthreads(); // Barrier for smem reuse

  int thread_expand_plan[ITEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_expand_plan[i] = std::min(int(ITEMS_PER_THREAD * threadIdx.x + i), num_tokens * p.topk);
  }

  BlockRadixSort(temp_storage.sort).Sort(thread_experts, thread_expand_plan);
  __syncthreads(); // Barrier for smem reuse

  BlockStore(temp_storage.store).Store(p.sorted_expert_indices, thread_experts, block_valid);
  __syncthreads(); // Barrier for smem reuse

  BlockStore(temp_storage.store).Store(p.expand_plan.mapping, thread_expand_plan, block_valid);
  __syncthreads(); // Barrier for smem reuse

  if (threadIdx.x == 0) {
    p.expand_plan.exclusive_sum[0] = 0;
  }
  if (threadIdx.x < (p.local_expert_end - p.local_expert_begin)) {
    int target = p.local_expert_begin + threadIdx.x;
    p.expand_plan.exclusive_sum[threadIdx.x + 1] = find_total_elts_leq_target(p.sorted_expert_indices, block_valid, target);
  }
}

//template <typename T, typename U, int BLOCK_SIZE, int COLS, std::enable_if_t<std::is_same_v<T, U>>* = nullptr>
//__global__ void expandForGemmKernel(ExpandForGemmKernelParams<T, U> p, int stage_idx)
template <typename T, typename U, int BLOCK_SIZE, int COLS>
__global__ typename std::enable_if<std::is_same<T, U>::value, void>::type expandForGemmKernel(ExpandForGemmKernelParams<T, U> p, int stage_idx)
{
  assert(BLOCK_SIZE == blockDim.x);

  int experts_per_stage = (p.local_expert_end - p.local_expert_begin) / p.num_stages;
  int begin = p.expand_plan.exclusive_sum[stage_idx == -1 ? 0 : (experts_per_stage * stage_idx)];
  int end = p.expand_plan.exclusive_sum[experts_per_stage * (stage_idx == -1 ? p.num_stages : (stage_idx + 1))];

  using VectorizedType = VectorizedType<T>;
  constexpr int ItemsPerVec = VectorizedType::ItemsPerVec;
  static_assert(COLS % ItemsPerVec == 0);

  for (int i = blockIdx.x; i < end - begin; i += gridDim.x)
  {
    const T *src = p.src + COLS * (p.expand_plan.mapping[begin + i] / p.topk);
    T *dst = p.dst + COLS * (begin + i);

#pragma unroll
    for (int k = threadIdx.x; k < COLS / ItemsPerVec; k += BLOCK_SIZE)
    {
      reinterpret_cast<VectorizedType *>(dst)[k] = reinterpret_cast<const VectorizedType *>(src)[k];
    }
  }
}

//template <typename T, typename U, int BLOCK_SIZE, int COLS, std::enable_if_t<!std::is_same_v<T, U>>* = nullptr>
//__global__ void expandForGemmKernel(ExpandForGemmKernelParams<T, U> p, int stage_idx)
template <typename T, typename U, int BLOCK_SIZE, int COLS>
__global__ typename std::enable_if<!std::is_same<T, U>::value, void>::type expandForGemmKernel(ExpandForGemmKernelParams<T, U> p, int stage_idx)
{
  assert(BLOCK_SIZE == blockDim.x);

  int experts_per_stage = (p.local_expert_end - p.local_expert_begin) / p.num_stages;
  int begin = p.expand_plan.exclusive_sum[stage_idx == -1 ? 0 : (experts_per_stage * stage_idx)];
  int end = p.expand_plan.exclusive_sum[experts_per_stage * (stage_idx == -1 ? p.num_stages : (stage_idx + 1))];

  using VectorizedTypeT = VectorizedType<T>;
  constexpr int ItemsPerVec = VectorizedTypeT::ItemsPerVec;
  static_assert(COLS % ItemsPerVec == 0);

  using VectorizedTypeU = VectorizedType<U>;

  for (int i = blockIdx.x; i < end - begin; i += gridDim.x)
  {
    const T *src = p.src + COLS * (p.expand_plan.mapping[begin + i] / p.topk);
    U *dst = p.dst + COLS * (begin + i);

#pragma unroll
    for (int k = threadIdx.x; k < COLS / ItemsPerVec; k += BLOCK_SIZE)
    {
      reinterpret_cast<VectorizedTypeU *>(dst)[k] = VectorizedTypeU{};
    }
  }
}

void getExpandForGemmPlan(ExpandForGemmPlanKernelParams p, cudaStream_t stream) {
  static constexpr int BLOCK_SIZE = 1024;
  static constexpr int ITEMS_PER_THREAD = 8;

  getExpandForGemmPlanKernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<1, BLOCK_SIZE, 0, stream>>>(p);
}

template <typename T, typename U>
void expandForGemm(ExpandForGemmKernelParams<T, U> p, int stage_idx, cudaStream_t stream) {
  int num_blocks;
  int block_size;
  auto num_blocks_heuristic = [&](int64_t num_tokens_hint, int topk, int num_stages){
    num_blocks = std::max(num_tokens_hint * topk / num_stages, 1L);
    block_size = 512;
    if (num_blocks > 256)
    {
      block_size = 256;
    }
    num_blocks = std::min(num_blocks, 1024);
  };
  num_blocks_heuristic(p.num_tokens_hint, p.topk, p.num_stages);

  BLOCK_SIZE_SWITCH(block_size,
    HIDDEN_SIZE_SWITCH(p.cols,
      [&]() {
        expandForGemmKernel<T, U, BLOCK_SIZE, HIDDEN_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(p, stage_idx);
      })
  )();
}

template <>
void expandForGemm<float, float>(ExpandForGemmKernelParams<float, float> p, int stage_idx, cudaStream_t stream) {
  int num_blocks;
  int block_size;
  auto num_blocks_heuristic = [&](int64_t num_tokens_hint, int topk, int num_stages){
    num_blocks = std::max(num_tokens_hint * topk / num_stages, 1L);
    block_size = 512;
    if (num_blocks > 256)
    {
      block_size = 256;
    }
    num_blocks = std::min(num_blocks, 1024);
  };
  num_blocks_heuristic(p.num_tokens_hint, p.topk, p.num_stages);

  BLOCK_SIZE_SWITCH(block_size,
    COLS_SWITCH(p.cols,
      [&]() {
        expandForGemmKernel<float, float, BLOCK_SIZE, COLS><<<num_blocks, BLOCK_SIZE, 0, stream>>>(p, stage_idx);
      })
  )();
}

template
void expandForGemm(ExpandForGemmKernelParams<float, float> p, int stage_idx, cudaStream_t stream);

template
void expandForGemm(ExpandForGemmKernelParams<float, __nv_fp8_e4m3> p, int stage_idx, cudaStream_t stream);

template
void expandForGemm(ExpandForGemmKernelParams<half, half> p, int stage_idx, cudaStream_t stream);

template
void expandForGemm(ExpandForGemmKernelParams<half, __nv_fp8_e4m3> p, int stage_idx, cudaStream_t stream);

#ifdef ENABLE_BF16
template
void expandForGemm(ExpandForGemmKernelParams<__nv_bfloat16, __nv_bfloat16> p, int stage_idx, cudaStream_t stream);

template
void expandForGemm(ExpandForGemmKernelParams<__nv_bfloat16, __nv_fp8_e4m3> p, int stage_idx, cudaStream_t stream);
#endif

template
void expandForGemm(ExpandForGemmKernelParams<__nv_fp8_e4m3, __nv_fp8_e4m3> p, int stage_idx, cudaStream_t stream);
} // namespace eps
