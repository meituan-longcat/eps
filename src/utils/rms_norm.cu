#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

#include "src/common/cuda_types.cuh"
#include "src/common/reduce_kernel_utils.cuh"

namespace eps {

template <typename T>
struct res_norm_ops_t {};

template <typename T>
struct res_norm_t {
  res_norm_ops_t<T> f;
  __device__ uint4 addvec(const uint4& a, const uint4& b, const uint4& bias, float& accum) const {
    uint4 c;
    c.x = f.cast(f.add(f.cast(a.x), f.cast(b.x), f.cast(bias.x), accum));
    c.y = f.cast(f.add(f.cast(a.y), f.cast(b.y), f.cast(bias.y), accum));
    c.z = f.cast(f.add(f.cast(a.z), f.cast(b.z), f.cast(bias.z), accum));
    c.w = f.cast(f.add(f.cast(a.w), f.cast(b.w), f.cast(bias.w), accum));
    return c;
  }
  __device__ uint4 normvec(const uint4& u, const uint4& s, float factor) const {
    uint4 v;
    v.x = f.cast(f.norm(f.cast(u.x), f.cast(s.x), factor));
    v.y = f.cast(f.norm(f.cast(u.y), f.cast(s.y), factor));
    v.z = f.cast(f.norm(f.cast(u.z), f.cast(s.z), factor));
    v.w = f.cast(f.norm(f.cast(u.w), f.cast(s.w), factor));
    return v;
  }
};

template <>
struct res_norm_ops_t<half> {
  __device__ float2 cast(const uint& x) const {
    return __half22float2(reinterpret_cast<const half2&>(x));
  }
  __device__ uint cast(const float2& x) const {
    auto y = __float22half2_rn(x);
    return reinterpret_cast<uint&>(y);
  }
  __device__ float2 add(const float2& a, const float2& b, const float2& bias, float& accum) const {
    float2 c{a.x + b.x + bias.x, a.y + b.y + bias.y};
    accum += c.x * c.x + c.y * c.y;
    return c;
  }
  __device__ float2 norm(const float2& a, const float2& s, float factor) const {
    return {a.x * s.x * factor, a.y * s.y * factor};
  }
};

template <>
struct res_norm_ops_t<float> {
  __device__ float cast(const uint& x) const {
    return reinterpret_cast<const float&>(x);
  }
  __device__ uint cast(const float& x) const {
    return reinterpret_cast<const uint&>(x);
  }
  __device__ float add(const float& a, const float& b, const float& bias, float& accum) const {
    float c = a + b + bias;
    accum += c * c;
    return c;
  }
  __device__ float norm(const float& a, const float& s, float factor) const {
    return a * s * factor;
  }
};

#ifdef ENABLE_BF16
template <>
struct res_norm_ops_t<__nv_bfloat16> {
  __device__ float2 cast(const uint& x) const {
    return cuda_cast<float2, __nv_bfloat162>(reinterpret_cast<const __nv_bfloat162&>(x));
  }
  __device__ uint cast(const float2& x) const {
    auto y = cuda_cast<__nv_bfloat162, float2>(x);
    return reinterpret_cast<uint&>(y);
  }
  __device__ float2 add(const float2& a, const float2& b, const float2& bias, float& accum) const {
    float2 c{a.x + b.x + bias.x, a.y + b.y + bias.y};
    accum += c.x * c.x + c.y * c.y;
    return c;
  }
  __device__ float2 norm(const float2& a, const float2& s, float factor) const {
    return {a.x * s.x * factor, a.y * s.y * factor};
  }
};
#endif

// fp16, bf16
// n is divided by 2 for this impl
template <typename T>
__global__ void rootMeanSquareNormKernel(T* out, const T* input, const T* scale, float eps, int m, int n) {
  using T2 = typename TypeConverter<T>::Type;
  __shared__ float s_inv_mean;
  float mean = 0.f;

  T2* out_ptr = (T2*)out;
  const T2* input_ptr = (const T2*)input;
  const T2* scale_ptr = (const T2*)scale;

  for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
    float2 tmp2 = cuda_cast<float2>(input_ptr[blockIdx.x * n + idx]);
    mean += tmp2.x * tmp2.x;
    mean += tmp2.y * tmp2.y;
  }

  mean = blockReduceSum<float>(mean);
  if (threadIdx.x == 0) {
    s_inv_mean = rsqrt(.5f * mean / (float)n + eps);
  }
  __syncthreads();

  for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
    float2 tmp2 = cuda_cast<float2>(input_ptr[blockIdx.x * n + idx]);
    float2 sca2 = cuda_cast<float2>(scale_ptr[idx]);
    tmp2.x = tmp2.x * s_inv_mean * sca2.x;
    tmp2.y = tmp2.y * s_inv_mean * sca2.y;
    out_ptr[blockIdx.x * n + idx] = cuda_cast<T2>(tmp2);
  }
}

template <typename T>
void rms_norm(T* out, const T* input, const T* scale, float eps, int m, int n, cudaStream_t stream) {
  static_assert(sizeof(T) == 2);

  if (n % 2 != 0) {
    throw std::runtime_error("dim " + std::to_string(n) + " is not divisible by 2.");
  }

  n /= 2;

  dim3 grid(m);
  dim3 block(std::min(n, 1024));
  rootMeanSquareNormKernel<<<grid, block, 0, stream>>>(out, input, scale, eps, m, n);
}

template void rms_norm(half*, const half*, const half*, float, int, int, cudaStream_t);
#ifdef ENABLE_BF16
template void
rms_norm(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float, int, int, cudaStream_t);
#endif

}
