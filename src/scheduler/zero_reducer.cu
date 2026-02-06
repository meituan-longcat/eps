#include <cub/cub.cuh>

#include "src/common/cuda_utils.cuh"
#include "src/scheduler/common.cuh"
#include "src/scheduler/utils.cuh"
#include "src/scheduler/zero_reducer.cuh"

namespace eps {

template <typename InputType, typename ExpertScaleType>
__global__ void zeroReducePlanKernel(ZeroReducePlanKernelParams<InputType, ExpertScaleType> p) {
  int tid = threadIdx.x;
  for (int token_idx = tid; token_idx < p.num_local_tokens; token_idx += blockDim.x) {
    p.zero_reduce_plan.zero_reduce_scales[token_idx] = 0;
  }
  for (int token_idx = tid; token_idx < p.num_local_tokens; token_idx += blockDim.x) {
    for (int k = 0; k < p.topk; ++k) {
      int kidx = token_idx * p.topk + k;
      if (p.expert_indices[kidx] < 0) {
        p.zero_reduce_plan.zero_reduce_scales[token_idx] += p.expert_scales[kidx];
      }
    }
  }
}

template <typename InputType, typename ExpertScaleType>
void getZeroReducePlan(ZeroReducePlanKernelParams<InputType, ExpertScaleType> p, cudaStream_t stream) {
  if (p.num_local_tokens <= 0) return;
  constexpr int BLOCK_SIZE = 512;
  zeroReducePlanKernel<<<1, BLOCK_SIZE, 0, stream>>>(p);
}

template <typename InputType, typename OutputType, typename ExpertScaleType>
__global__ void zeroReduceKernel(ZeroReduceKernelParams<InputType, OutputType, ExpertScaleType> p) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int token_idx = idx / p.hidden_size;
  if (token_idx >= p.num_local_tokens) return;
  const int hidden_idx = idx % p.hidden_size;
  if (hidden_idx >= p.hidden_size) return;
  p.output[idx] +=
      cuda_cast<OutputType>(cuda_cast<float>(p.raw_input[idx]) * cuda_cast<float>(p.zero_reduce_scales[token_idx]));
}

template <typename InputType, typename OutputType, typename ExpertScaleType>
void zeroReduce(ZeroReduceKernelParams<InputType, OutputType, ExpertScaleType> p, cudaStream_t stream) {
  if (p.num_local_tokens <= 0) return;
  constexpr int BLOCK_SIZE = 512;
  int num_blocks = CEILDIV(p.num_local_tokens * p.hidden_size, BLOCK_SIZE);
  zeroReduceKernel<InputType, OutputType, ExpertScaleType><<<num_blocks, BLOCK_SIZE, 0, stream>>>(p);
}

template void zeroReduce(ZeroReduceKernelParams<float, float, float> p, cudaStream_t stream);
template void zeroReduce(ZeroReduceKernelParams<half, half, float> p, cudaStream_t stream);
template void zeroReduce(ZeroReduceKernelParams<half, half, half> p, cudaStream_t stream);
template void zeroReduce(ZeroReduceKernelParams<__nv_bfloat16, __nv_bfloat16, float> p, cudaStream_t stream);
template void zeroReduce(ZeroReduceKernelParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16> p, cudaStream_t stream);
template void zeroReduce(ZeroReduceKernelParams<__nv_fp8_e4m3, float, float> p, cudaStream_t stream);
template void zeroReduce(ZeroReduceKernelParams<__nv_fp8_e4m3, half, float> p, cudaStream_t stream);
template void zeroReduce(ZeroReduceKernelParams<__nv_fp8_e4m3, half, half> p, cudaStream_t stream);
template void zeroReduce(ZeroReduceKernelParams<__nv_fp8_e4m3, __nv_bfloat16, float> p, cudaStream_t stream);
template void zeroReduce(ZeroReduceKernelParams<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16> p, cudaStream_t stream);

template void getZeroReducePlan(ZeroReducePlanKernelParams<float, float> p, cudaStream_t stream);
template void getZeroReducePlan(ZeroReducePlanKernelParams<half, float> p, cudaStream_t stream);
template void getZeroReducePlan(ZeroReducePlanKernelParams<half, half> p, cudaStream_t stream);
template void getZeroReducePlan(ZeroReducePlanKernelParams<__nv_bfloat16, float> p, cudaStream_t stream);
template void getZeroReducePlan(ZeroReducePlanKernelParams<__nv_bfloat16, __nv_bfloat16> p, cudaStream_t stream);
template void getZeroReducePlan(ZeroReducePlanKernelParams<__nv_fp8_e4m3, float> p, cudaStream_t stream);
template void getZeroReducePlan(ZeroReducePlanKernelParams<__nv_fp8_e4m3, half> p, cudaStream_t stream);
template void getZeroReducePlan(ZeroReducePlanKernelParams<__nv_fp8_e4m3, __nv_bfloat16> p, cudaStream_t stream);

}  // namespace eps
