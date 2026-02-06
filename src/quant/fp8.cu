#include "src/quant/fp8.cuh"

#include <cmath>

#include <cooperative_groups.h>

#include <cub/util_type.cuh>
#include <cub/cub.cuh>

#include "src/common/debug.cuh"
#include "src/common/cuda_utils.cuh"

namespace cg = cooperative_groups;

namespace eps {

// __host__ __device__ constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();
// constexpr static float FP8_E4M3_MAX = 448.0f;

__global__ void allReduceMaxScale(AllReduceMaxScaleKernelParams p)
{
  /*
  int warpSize = 32;
  int warpIdx = threadIdx.x / warpSize;
  int laneIdx = threadIdx.x % warpSize;
  if (warpIdx < p.ep_world_size)
  {
    int remoteRank = warpIdx;
    auto smChan = p.smChans[remoteRank];
    smChan.put<sizeof(float)>(
        p.my_rank * sizeof(float),
        0,
        sizeof(float),
        laneIdx,
        warpSize);
    if (laneIdx == 0)
    {
      smChan.signal();
      smChan.wait();
    }
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    float max = 0.0f;
    for (int rank = 0; rank < p.ep_world_size; rank++)
    {
      if (p.gathered_scales[rank] > max)
      {
        max = p.gathered_scales[rank];
      }
    }
    *p.scale = max;
  }
  */
}

namespace _vllm
{

  __device__ __forceinline__ float atomicMaxFloat(float *addr, float value)
  {
    float old;
    old = (value >= 0)
              ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
              : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
  }

  // Compute the absolute maximum m of the input tensor and store
  // m / float8_e4m3::max() in *scale. Each thread block performs a
  // reduction tree and the memory in scale is atomically updated.
  // So to get the right answer, *scale needs to be initialized to
  // a value <= 0.0 and we need to wait for all thread blocks to
  // finish before consuming *scale.
  template <typename scalar_t>
  __global__ void segmented_max_reduction(float *__restrict__ scale,
                                          const scalar_t *__restrict__ input,
                                          int64_t num_elems)
  {
    __shared__ float cache[1024];
    int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    // First store maximum for all values processes by
    // the current thread in cache[threadIdx.x]
    float tmp = 0.0f;
    while (i < num_elems)
    {
      float x = static_cast<float>(input[i]);
      tmp = max(tmp, fabs(x));
      i += blockDim.x * gridDim.x;
    }
    cache[threadIdx.x] = tmp;

    __syncthreads();

    // Now perform parallel reduction within the thread block
    int ib = blockDim.x / 2;
    while (ib != 0)
    {
      if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x])
      {
        cache[threadIdx.x] = cache[threadIdx.x + ib];
      }
      __syncthreads();
      ib /= 2;
    }
    // Finally, since cache[0] contains the maximum for this thread block,
    // atomically write the max to the target location
    if (threadIdx.x == 0)
    {
      atomicMaxFloat(scale, cache[0] / FP8_E4M3_MAX);
    }
  }

  template <typename scalar_t>
  struct __align__(8) vec4_t
  {
    scalar_t x;
    scalar_t y;
    scalar_t z;
    scalar_t w;
  };

  typedef struct __align__(4)
  {
    FP8_TYPE x;
    FP8_TYPE y;
    FP8_TYPE z;
    FP8_TYPE w;
  }
  float8x4_t;

  template <typename scalar_t>
  __device__ float thread_max_vec(scalar_t const *__restrict__ input,
                                  int64_t const num_elems, int const tid,
                                  int const step)
  {
    // Vectorized input/output to better utilize memory bandwidth.
    vec4_t<scalar_t> const *vectorized_in =
        reinterpret_cast<vec4_t<scalar_t> const *>(input);

    int64_t const num_vec_elems = num_elems >> 2;
    float absmax_val = 0.0f;

#pragma unroll 4
    for (int64_t i = tid; i < num_vec_elems; i += step)
    {
      vec4_t<scalar_t> in_vec = vectorized_in[i];
      absmax_val = max(absmax_val, fabs(in_vec.x));
      absmax_val = max(absmax_val, fabs(in_vec.y));
      absmax_val = max(absmax_val, fabs(in_vec.z));
      absmax_val = max(absmax_val, fabs(in_vec.w));
    }

    // Handle the remaining elements if num_elems is not divisible by 4
    for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step)
    {
      absmax_val = max(absmax_val, fabs(input[i]));
    }

    return absmax_val;
  }

  template <bool is_scale_inverted>
  __device__ __forceinline__ FP8_TYPE scaled_fp8_conversion(float const val, float const scale)
  {
    float x = 0.0f;
    if constexpr (is_scale_inverted)
    {
      x = val * scale;
    }
    else
    {
      x = val / scale;
    }

    float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
    return static_cast<FP8_TYPE>(r);
  }

  template <typename scalar_t, bool is_scale_inverted>
  __device__ void scaled_fp8_conversion_vec(FP8_TYPE *__restrict__ out,
                                            scalar_t const *__restrict__ input,
                                            float const scale,
                                            int64_t const num_elems,
                                            int const tid, int const step)
  {
    // Vectorized input/output to better utilize memory bandwidth.
    vec4_t<scalar_t> const *vectorized_in =
        reinterpret_cast<vec4_t<scalar_t> const *>(input);
    float8x4_t *vectorized_out = reinterpret_cast<float8x4_t *>(out);

    int64_t const num_vec_elems = num_elems >> 2;

#pragma unroll 4
    for (int64_t i = tid; i < num_vec_elems; i += step)
    {
      vec4_t<scalar_t> in_vec = vectorized_in[i];
      float8x4_t out_vec;

      out_vec.x = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(in_vec.x), scale);
      out_vec.y = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(in_vec.y), scale);
      out_vec.z = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(in_vec.z), scale);
      out_vec.w = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(in_vec.w), scale);
      vectorized_out[i] = out_vec;
    }

    // Handle the remaining elements if num_elems is not divisible by 4
    for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step)
    {
      out[i] = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(input[i]), scale);
    }
  }

  template <typename scalar_t>
  __global__ void scaled_fp8_quant_kernel(FP8_TYPE *__restrict__ out,
                                          const scalar_t *__restrict__ input,
                                          const float *__restrict__ scale,
                                          int64_t num_elems)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Invert the scale so that we can use multiplications to avoid expensive
    // division.
    const float inverted_scale = 1.0f / (*scale);
    scaled_fp8_conversion_vec<scalar_t, true>(out, input, inverted_scale, num_elems, tid, blockDim.x * gridDim.x);
  }
} // namespace _vllm

template<int N>
struct float8xn_t;

template<>
struct float8xn_t<8> {
  uint64_t data;
};

template <typename T, int GROUP_SIZE>
__device__ void per_token_group_quant_fp8_impl(const T* input, int cols, int num_tokens, float* scales, FP8_TYPE* output)
{
    constexpr static int thread_group_size = GROUP_SIZE / (VectorizedType<T>::ItemsPerVec);
    // auto tile = cg::tiled_partition(cg::this_thread_block(), thread_group_size);
    auto tile = cg::tiled_partition<thread_group_size>(cg::this_thread_block());

    int num_thread_groups = blockDim.x / thread_group_size;
    int thread_group_idx = threadIdx.x / thread_group_size;
    using ResultType = float8xn_t<VectorizedType<T>::ItemsPerVec>;

    for (int i = blockIdx.x; i < num_tokens; i += gridDim.x) {
      for (int j = thread_group_idx; j < cols / GROUP_SIZE; j += num_thread_groups)
      {
        VectorizedType<T> v = *(reinterpret_cast<const VectorizedType<T> *>(input + i * cols + j * GROUP_SIZE) + tile.thread_rank());
        float absmax = v.absmax();

        for (int offset = tile.size() / 2; offset > 0; offset /= 2)
        {
          absmax = max(absmax, tile.shfl_xor(absmax, offset));
        }

        // float scale = absmax / FP8_E4M3_MAX;
        // 会被编译为 div.rn.f32 %f1, %f34, 0f43E00000;
        // 以下使用 ptx 目的是为了和 triton 版本数值上严格对齐。
        float scale;
        asm("div.full.f32 %0, %1, %2;"
        : "=f"(scale)
        : "f"(absmax), "f"(FP8_E4M3_MAX));

        if (tile.thread_rank() == 0)
        {
          float *scales_ptr = scales + i * cols / GROUP_SIZE + j;
          *scales_ptr = scale;
        }

        float reverse_scale = 1 / scale;
        ResultType result{};
#pragma unroll
        for (int k = 0; k < VectorizedType<T>::ItemsPerVec; k++)
        {
          T *data = (T *)&v.data;
          float x = static_cast<float>(data[k]) * reverse_scale;
          float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
          reinterpret_cast<FP8_TYPE*>(&result.data)[k] = static_cast<FP8_TYPE>(r);
        }

        *(reinterpret_cast<ResultType *>(output + i * cols + j * GROUP_SIZE) + tile.thread_rank()) = result;
      }
    }
}

template <typename T, int GROUP_SIZE>
__device__ void per_token_group_dequant_fp8_impl(T* output, int cols, int num_tokens, float* scales, const FP8_TYPE* input)
{
    constexpr static int thread_group_size = GROUP_SIZE / (VectorizedType<T>::ItemsPerVec);
    // auto tile = cg::tiled_partition(cg::this_thread_block(), thread_group_size);
    auto tile = cg::tiled_partition<thread_group_size>(cg::this_thread_block());

    int num_thread_groups = blockDim.x / thread_group_size;
    int thread_group_idx = threadIdx.x / thread_group_size;

    using ResultType = VectorizedType<T>;
    using InputType = float8xn_t<VectorizedType<T>::ItemsPerVec>;

    for (int i = blockIdx.x; i < num_tokens; i += gridDim.x) {
      for (int j = thread_group_idx; j < cols / GROUP_SIZE; j += num_thread_groups)
      {
        float scale = scales[i * cols / GROUP_SIZE + j];

        InputType quanted = *(reinterpret_cast<const InputType *>(input + i * cols + j * GROUP_SIZE) + tile.thread_rank());

        ResultType result{};
        #pragma unroll
        for (int k = 0; k < VectorizedType<T>::ItemsPerVec; k++)
        {
          FP8_TYPE v = *(reinterpret_cast<FP8_TYPE*>(&quanted.data) + k);
          float x = static_cast<float>(v) * scale;
          *(reinterpret_cast<T*>(&result) + k) = x;
        }

        *(reinterpret_cast<ResultType *>(output + i * cols + j * GROUP_SIZE) + tile.thread_rank()) = result;
      }
    }
}

// GROUP_SIZE % 32 == 0
// cols % GROUP_SIZE == 0
template <typename T, int GROUP_SIZE>
__global__ void per_token_group_quant_fp8_kernel(PerTokenGroupQuantFP8Params<T> p)
{
  int begin = p.exclusive_sum[0];
  int end = p.exclusive_sum[p.num_experts];

  T* origin = p.origin + begin * p.cols;
  float* scales = p.scales + begin * p.cols / GROUP_SIZE;
  FP8_TYPE* quanted = p.quanted + begin * p.cols;

  if (p.dequant) {
    per_token_group_dequant_fp8_impl<T, GROUP_SIZE>(origin, p.cols, end - begin, scales, quanted);
  } else {
    per_token_group_quant_fp8_impl<T, GROUP_SIZE>(origin, p.cols, end - begin, scales, quanted);
  }
}

template <typename T, int GROUP_SIZE>
__global__ void per_token_group_quant_fp8_kernel(PerTokenGroupQuantFP8DenseParams<T> p)
{
  if (p.dequant) {
    per_token_group_dequant_fp8_impl<T, GROUP_SIZE>(p.origin, p.cols, p.num_tokens, p.scales, p.quanted);
  } else {
    per_token_group_quant_fp8_impl<T, GROUP_SIZE>(p.origin, p.cols, p.num_tokens, p.scales, p.quanted);
  }
}

template <typename T, template<typename> typename Params>
void per_token_group_quant_fp8(Params<T> params, cudaStream_t stream)
{
  if (params.group_size % 32 != 0) {
    throw std::runtime_error("group_size: " + std::to_string(params.group_size) + " is not divisible by 32.");
  }

  if (params.cols % params.group_size != 0) {
    throw std::runtime_error("cols: " + std::to_string(params.cols) + " is not divisible by group_size: " + std::to_string(params.group_size));
  }

  if (params.group_size == 128)
  {
    constexpr static int thread_group_size = 128 / (VectorizedType<T>::ItemsPerVec);
    int block_size = std::min<int>(params.cols / params.group_size * thread_group_size, 256);
    per_token_group_quant_fp8_kernel<T, 128><<<params.num_thread_blocks(), block_size, 0, stream>>>(params);
  }
  else
  {
    throw std::runtime_error("Not supported group size: " + std::to_string(params.group_size));
  }
}

template
void per_token_group_quant_fp8<half, PerTokenGroupQuantFP8Params>(PerTokenGroupQuantFP8Params<half> params, cudaStream_t stream);

template
void per_token_group_quant_fp8<__nv_bfloat16, PerTokenGroupQuantFP8Params>(PerTokenGroupQuantFP8Params<__nv_bfloat16> params, cudaStream_t stream);

template<>
void per_token_group_quant_fp8<float, PerTokenGroupQuantFP8Params>(PerTokenGroupQuantFP8Params<float> params, cudaStream_t stream) {
  throw std::runtime_error("per_token_group_quant_fp8 float is not supported.");
}

template
void per_token_group_quant_fp8<half, PerTokenGroupQuantFP8DenseParams>(PerTokenGroupQuantFP8DenseParams<half> params, cudaStream_t stream);

template
void per_token_group_quant_fp8<__nv_bfloat16, PerTokenGroupQuantFP8DenseParams>(PerTokenGroupQuantFP8DenseParams<__nv_bfloat16> params, cudaStream_t stream);

template<>
void per_token_group_quant_fp8<float, PerTokenGroupQuantFP8DenseParams>(PerTokenGroupQuantFP8DenseParams<float> params, cudaStream_t stream) {
  throw std::runtime_error("per_token_group_quant_fp8 float is not supported.");
}

template <typename scalar_t>
void static_scaled_fp8_quant(FP8_TYPE *out,         // [..., d]
                             const scalar_t *input, // [..., d]
                             const float *scale,    // [1]
                             size_t num_elems,
                             size_t num_cols,
                             cudaStream_t stream)
{
  dim3 grid(num_elems / num_cols);
  dim3 block(1024);
  _vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
      out, input, scale, num_elems);
}

template <typename scalar_t>
void dynamic_scaled_fp8_quant(FP8_TYPE *out,         // [..., d]
                              const scalar_t *input, // [..., d]
                              float *scale,          // [1]
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream)
{
  dim3 grid(num_elems / num_cols);
  dim3 block(1024);
  cudaMemsetAsync(scale, 0, sizeof(float), stream);
  _vllm::segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
      scale, input, num_elems);
  _vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
      out, input, scale, num_elems);
}

template <typename scalar_t>
void dynamic_scaled_fp8_quant(FP8_TYPE *out,         // [..., d]
                              const scalar_t *input, // [..., d]
                              AllReduceMaxScaleKernelParams p,
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream)
{
  dim3 grid(num_elems / num_cols);
  dim3 block(1024);
  cudaMemsetAsync(p.scale, 0, sizeof(float), stream);
  _vllm::segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
      p.scale, input, num_elems);
  allReduceMaxScale<<<1, 32 * p.ep_world_size, 0, stream>>>(p);
  _vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
      out, input, p.scale, num_elems);
}

template
void static_scaled_fp8_quant<float>(FP8_TYPE *out,         // [..., d]
                             const float *input, // [..., d]
                             const float *scale,    // [1]
                             size_t num_elems,
                             size_t num_cols,
                             cudaStream_t stream);

template
void dynamic_scaled_fp8_quant<float>(FP8_TYPE *out,         // [..., d]
                              const float *input, // [..., d]
                              float *scale,          // [1]
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream);

template
void dynamic_scaled_fp8_quant<float>(FP8_TYPE *out,         // [..., d]
                              const float *input, // [..., d]
                              AllReduceMaxScaleKernelParams p,
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream);

template
void static_scaled_fp8_quant<half>(FP8_TYPE *out,         // [..., d]
                             const half *input, // [..., d]
                             const float *scale,    // [1]
                             size_t num_elems,
                             size_t num_cols,
                             cudaStream_t stream);

template
void dynamic_scaled_fp8_quant<half>(FP8_TYPE *out,         // [..., d]
                              const half *input, // [..., d]
                              float *scale,          // [1]
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream);

template
void dynamic_scaled_fp8_quant<half>(FP8_TYPE *out,         // [..., d]
                              const half *input, // [..., d]
                              AllReduceMaxScaleKernelParams p,
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream);

template
void static_scaled_fp8_quant<__nv_bfloat16>(FP8_TYPE *out,         // [..., d]
                             const __nv_bfloat16 *input, // [..., d]
                             const float *scale,    // [1]
                             size_t num_elems,
                             size_t num_cols,
                             cudaStream_t stream);

template
void dynamic_scaled_fp8_quant<__nv_bfloat16>(FP8_TYPE *out,         // [..., d]
                              const __nv_bfloat16 *input, // [..., d]
                              float *scale,          // [1]
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream);

template
void dynamic_scaled_fp8_quant<__nv_bfloat16>(FP8_TYPE *out,         // [..., d]
                              const __nv_bfloat16 *input, // [..., d]
                              AllReduceMaxScaleKernelParams p,
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream);

}