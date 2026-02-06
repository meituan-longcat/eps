#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include "src/common/cuda_fp8_utils.h"

namespace eps {

#ifdef ENABLE_BF16
inline __host__ __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}

inline __device__ int16_t bf1622int16(__nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = max(min(__low2float(val), 127.f), -128.f);
  f_val.y = max(min(__high2float(val), 127.f), -128.f);
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int8[0] = static_cast<int8_t>(static_cast<short>(f_val.x));
  int8[1] = static_cast<int8_t>(static_cast<short>(f_val.y));
  return int16;
#else
  val = __hmin2(val, make_bfloat162(127., 127.));
  val = __hmax2(val, make_bfloat162(-128., -128.));
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int8[0] = static_cast<int8_t>(static_cast<short>(val.x));
  int8[1] = static_cast<int8_t>(static_cast<short>(val.y));
  return int16;
#endif
}

inline __device__ __nv_bfloat162 float22bf162(const float2 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __floats2bfloat162_rn(val.x, val.y);
#else
  return __float22bfloat162_rn(val);
#endif
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __nv_bfloat162 val2;
  val2.x = val;
  val2.y = val;
  return val2;
#else
  return __bfloat162bfloat162(val);
#endif
}

inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#else
  return __hadd2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) + __bfloat162float(y));
#else
  return __hadd(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hsub2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  return __floats2bfloat162_rn(fxl - fyl, fxh - fyh);
#else
  return __hsub2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hsub(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) - __bfloat162float(y));
#else
  return __hsub(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  return __floats2bfloat162_rn(fxl * fyl, fxh * fyh);
#else
  return __hmul2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y));
#else
  return __hmul(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hfma2(const __nv_bfloat162 x, const __nv_bfloat162 y, const __nv_bfloat162 z) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh, fzl, fzh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  fzl = __low2float(z);
  fzh = __high2float(z);
  return __floats2bfloat162_rn(fxl * fyl + fzl, fxh * fyh + fzh);
#else
  return __hfma2(x, y, z);
#endif
}

inline __device__ __nv_bfloat16 bf16hfma(const __nv_bfloat16 x, const __nv_bfloat16 y, const __nv_bfloat16 z) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y) + __bfloat162float(z));
#else
  return __hfma(x, y, z);
#endif
}

inline __device__ __nv_bfloat162 bf16exp2(const __nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  ;
  return __floats2bfloat162_rn(expf(fxl), expf(fxh));
#else
  return h2exp(x);
#endif
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
inline __device__ __nv_bfloat162 operator*(const __nv_bfloat162 x, const __nv_bfloat162 y) {
  return bf16hmul2(x, y);
};
inline __device__ __nv_bfloat162 operator+(const __nv_bfloat162 x, const __nv_bfloat162 y) {
  return bf16hadd2(x, y);
};

inline __device__ __nv_bfloat162 make_bfloat162(const __nv_bfloat16 x, const __nv_bfloat16 y) {
  __nv_bfloat162 t;
  t.x = x;
  t.y = y;
  return t;
}

#endif

inline __device__ __nv_bfloat16 bf16hadd(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b) + __bfloat162float(c));
#else
  return a + b + c;
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b) + __bfloat162float(c) + __bfloat162float(d));
#else
  return (__nv_bfloat16)((float)a + (float)b + (float)c + (float)d);
#endif
}

inline __device__ __nv_bfloat162 bf16hadd2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fal, fah, fbl, fbh, fcl, fch;
  fal = __low2float(a);
  fah = __high2float(a);
  fbl = __low2float(b);
  fbh = __high2float(b);
  fcl = __low2float(c);
  fch = __high2float(c);
  return __floats2bfloat162_rn(fal + fbl + fcl, fah + fbh + fch);
#else
  return a + b + c;
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) * __bfloat162float(c));
#else
  return a * b * c;
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fal, fah, fbl, fbh, fcl, fch;
  fal = __low2float(a);
  fah = __high2float(a);
  fbl = __low2float(b);
  fbh = __high2float(b);
  fcl = __low2float(c);
  fch = __high2float(c);
  return __floats2bfloat162_rn(fal * fbl * fcl, fah * fbh * fch);
#else
  return a * b * c;
#endif
}

inline __device__ __nv_bfloat162 bf16hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c, __nv_bfloat162 d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fal, fah, fbl, fbh, fcl, fch, fdl, fdh;
  fal = __low2float(a);
  fah = __high2float(a);
  fbl = __low2float(b);
  fbh = __high2float(b);
  fcl = __low2float(c);
  fch = __high2float(c);
  fdl = __low2float(d);
  fdh = __high2float(d);
  return __floats2bfloat162_rn(fal * fbl * fcl + fdl, fah * fbh * fch + fdh);
#else
  return a * b * c + d;
#endif
}

#endif // ENABLE_BF16

template <typename T>
inline __device__ T ldg(const T* val) {
  return __ldg(val);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 ldg(const __nv_bfloat162* val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return val[0];
#else
  return __ldg(val);
#endif
}

template <>
inline __device__ __nv_bfloat16 ldg(const __nv_bfloat16* val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return val[0];
#else
  return __ldg(val);
#endif
}
#endif // ENABLE_BF16

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
}; // keep for generality

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

#if ENABLE_BF16
template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};
#endif // ENABLE_BF16

// Defined math operations (bfloat16 fallback to fp32 when it is not supported)
template <typename T>
inline __device__ T hadd2(T a, T b) {
  return __hadd2(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
  return bf16hadd2(a, b);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T add(T a, T b) {
  return a + b;
}

template <>
inline __device__ half2 add(half2 a, half2 b) {
  return __hadd2(a, b);
}

template <>
inline __device__ half add(half a, half b) {
  return __hadd(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
  return bf16hadd2(a, b);
}

template <>
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
  return bf16hadd(a, b);
}

inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, float b) {
  return bf16hadd(a, __float2bfloat16(b));
}
#endif // ENABLE_BF16

// applies to all 4 values addition
template <typename T>
inline __device__ T add(T a, T b, T c) {
  return a + b + c;
}

#if ENABLE_BF16
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
  return bf16hadd(a, b, c);
}

inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
  return bf16hadd2(a, b, c);
}
#endif // ENABLE_BF16

// applies to all 4 values addition
template <typename T>
inline __device__ T add(T a, T b, T c, T d) {
  return (T)((float)a + (float)b + (float)c + (float)d);
}

#if ENABLE_BF16
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d) {
  return bf16hadd(a, b, c, d);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T hsub2(T a, T b) {
  return __hsub2(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
  return bf16hsub2(a, b);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T hmul2(T a, T b) {
  return __hmul2(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
  return bf16hmul2(a, b);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T hmul2(T a, T b, T c) {
  return a * b * c;
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hmul2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
  return bf16hmul2(a, b, c);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T mul(T a, T b, T c) {
  return a * b * c;
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
  return bf16hmul(a, b, c);
}

inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
  return bf16hmul2(a, b, c);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T fma(T a, T b, T c, T d) {
  return a * b * c + d;
}

#if ENABLE_BF16
inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c, __nv_bfloat162 d) {
  return bf16hfma2(a, b, c, d);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T fma(T a, T b, T c) {
  return a * b + c;
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
  return bf16hfma2(a, b, c);
}

template <>
inline __device__ __nv_bfloat16 fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
  return bf16hfma(a, b, c);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T hexp2(T a) {
  return h2exp(a);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hexp2(__nv_bfloat162 a) {
  return bf16exp2(a);
}
#endif // ENABLE_BF16

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float2 cuda_cast<float2, int2>(int2 val) {
  return make_float2(val.x, val.y);
}
template <>
__device__ inline float2 cuda_cast<float2, float>(float val) {
  return make_float2(val, val);
}
template <>
__device__ inline float2 cuda_cast<float2, half2>(half2 val) {
  return __half22float2(val);
}
template <>
__device__ inline half2 cuda_cast<half2, float2>(float2 val) {
  return __float22half2_rn(val);
}
template <>
__device__ inline half2 cuda_cast<half2, float>(float val) {
  return __float2half2_rn(val);
}
template <>
__device__ inline half2 cuda_cast<half2, half>(half val) {
  return __half2half2(val);
}

template <>
__device__ inline int8_t cuda_cast<int8_t, half>(half val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  union {
    half fp16;
    int16_t int16_in;
  };
  fp16 = val;
  asm volatile("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(int16) : "h"(int16_in));
  return int8[0];
}

template <>
__device__ inline int16_t cuda_cast<int16_t, half2>(half2 val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int8[0] = cuda_cast<int8_t>(val.x);
  int8[1] = cuda_cast<int8_t>(val.y);
  return int16;
}

template <>
__device__ inline int8_t cuda_cast<int8_t, float>(float val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
  return int8[0];
}

template <>
__device__ inline int16_t cuda_cast<int16_t, float2>(float2 val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int8[0] = cuda_cast<int8_t>(val.x);
  int8[1] = cuda_cast<int8_t>(val.y);
  return int16;
}

template <>
__device__ inline half2 cuda_cast<half2, int16_t>(int16_t val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int16 = val;
  return make_half2(int8[0], int8[1]);
}

template <>
__device__ inline float2 cuda_cast<float2, int16_t>(int16_t val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int16 = val;
  return make_float2(int8[0], int8[1]);
}

template <>
__device__ inline uint32_t cuda_cast<uint32_t, float2>(float2 val) {
  union {
    uint32_t uint32;
    half2 fp162;
  };
  fp162.x = static_cast<half>(val.x);
  fp162.y = static_cast<half>(val.y);
  return uint32;
}
template <>
__device__ inline float2 cuda_cast<float2, uint32_t>(uint32_t val) {
  union {
    uint32_t uint32;
    half2 fp162;
  };
  uint32 = val;
  return cuda_cast<float2>(fp162);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat16 cuda_cast(int32_t val) {
  return static_cast<float>(val);
}
template <>
__device__ inline __nv_bfloat16 cuda_cast(int8_t val) {
  return static_cast<float>(val);
}
template <>
__device__ inline int8_t cuda_cast(__nv_bfloat16 val) {
  return static_cast<float>(val);
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <>
__device__ inline float2 cuda_cast<float2, __nv_bfloat162>(__nv_bfloat162 val) {
  return bf1622float2(val);
}

template <>
__device__ inline half cuda_cast<half, __nv_bfloat16>(__nv_bfloat16 val) {
  return __float2half(__bfloat162float(val));
}

template <>
__device__ inline int16_t cuda_cast<int16_t, __nv_bfloat162>(__nv_bfloat162 val) {
  return bf1622int16(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}
template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, half>(half val) {
  return __float2bfloat16(__half2float(val));
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_bfloat16>(__nv_bfloat16 val) {
  return bf162bf162(val);
}
template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float>(float val) {
  return __float2bfloat162_rn(val);
}
template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float2>(float2 val) {
  return float22bf162(val);
}
template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, int16_t>(int16_t val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int16 = val;
  __nv_bfloat162 res;
  res.x = cuda_cast<__nv_bfloat16>(int8[0]);
  res.y = cuda_cast<__nv_bfloat16>(int8[1]);
  return res;
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, half2>(half2 val) {
  return float22bf162(__half22float2(val));
}

#endif // ENABLE BF16

template <typename T>
__device__ inline T cuda_abs(T val);
template <>
__device__ inline float cuda_abs(float val) {
  return fabs(val);
}
template <>
__device__ inline half cuda_abs(half val) {
  return __habs(val);
}
template <>
__device__ inline half2 cuda_abs(half2 val) {
  return __habs2(val);
}

template <>
__device__ inline float2 cuda_abs(float2 val) {
  return make_float2(fabs(val.x), fabs(val.y));
}

#ifdef ENABLE_BF16

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val) {
  return __habs(val);
}
template <>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val) {
  return __habs2(val);
}
#else
template <>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val) {
  return fabs(cuda_cast<float>(val));
}
template <>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val) {
  return make_bfloat162(fabs(cuda_cast<float>(val.x)), fabs(cuda_cast<float>(val.y)));
}
#endif

#endif // ENABLE_FP16

// Unary maximum: compute the max of a vector type
template <typename To, typename Ti>
__device__ inline To cuda_max(Ti val) {
  return cuda_cast<To>(val);
};

template <>
__device__ inline float cuda_max(float2 val) {
  return fmaxf(val.x, val.y);
}

template <>
__device__ inline half cuda_max(half2 val) {
  return __hmax(val.x, val.y);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat16 cuda_max(__nv_bfloat162 val) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  return __hmax(val.x, val.y);
#else
  asm volatile("brkpt;\n" ::);
  return __nv_bfloat16();
#endif
}
#endif

// Binary maximum: compute the max of two scalar types
template <typename T>
__device__ inline T cuda_max(T val1, T val2) {
  return (val1 > val2) ? val1 : val2;
}

template <>
__device__ inline float2 cuda_max(float2 val1, float2 val2) {
  float2 out;
  out.x = fmaxf(val1.x, val2.x);
  out.y = fmaxf(val1.y, val2.y);
  return out;
}

template <>
__device__ inline half2 cuda_max(half2 val1, half2 val2) {
  return __hmax2(val1, val2);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat162 cuda_max(__nv_bfloat162 val1, __nv_bfloat162 val2) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  return __hmax2(val1, val2);
#else
  asm volatile("brkpt;\n" ::);
  return __nv_bfloat162();
#endif
}
#endif // ENABLE_BF16

#ifdef ENABLE_FP8
template <>
__device__ inline float cuda_cast<float, __nv_fp8_e4m3>(__nv_fp8_e4m3 val) {
  return __half2float(fp8_e4m3_to_half(val));
}
template <>
__device__ inline half cuda_cast<half, __nv_fp8_e4m3>(__nv_fp8_e4m3 val) {
  return fp8_e4m3_to_half(val);
}
template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, __nv_fp8_e4m3>(__nv_fp8_e4m3 val) {
  return __float2bfloat16(__half2float(fp8_e4m3_to_half(val)));
}
template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, float>(float val) {
  return float_to_fp8_e4m3(val);
}
template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, half>(half val) {
  return half_to_fp8_e4m3(val);
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, __nv_bfloat16>(__nv_bfloat16 val) {
  return bfloat16_to_fp8_e4m3(val);
}

template <>
__device__ inline int8_t cuda_cast<int8_t, __nv_fp8_e4m3>(__nv_fp8_e4m3 val) {
  // no impl
  return 0;
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, int8_t>(int8_t val) {
  return cuda_cast<__nv_fp8_e4m3>(cuda_cast<__nv_bfloat16>(cuda_cast<float>(val)));
}
// x2
template <>
__device__ inline float2 cuda_cast<float2, __nv_fp8x2_e4m3>(__nv_fp8x2_e4m3 val) {
  return __half22float2(fp8x2_e4m3_to_half2(val));
}
template <>
__device__ inline half2 cuda_cast<half2, __nv_fp8x2_e4m3>(__nv_fp8x2_e4m3 val) {
  return fp8x2_e4m3_to_half2(val);
}
template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_fp8x2_e4m3>(__nv_fp8x2_e4m3 val) {
  return __float22bfloat162_rn(__half22float2(fp8x2_e4m3_to_half2(val)));
}
template <>
__device__ inline __nv_fp8x2_e4m3 cuda_cast<__nv_fp8x2_e4m3, float2>(float2 val) {
  return float2_to_fp8x2_e4m3(val);
}
template <>
__device__ inline __nv_fp8x2_e4m3 cuda_cast<__nv_fp8x2_e4m3, half2>(half2 val) {
  return half2_to_fp8x2_e4m3(val);
}
template <>
__device__ inline __nv_fp8x2_e4m3 cuda_cast<__nv_fp8x2_e4m3, __nv_bfloat162>(__nv_bfloat162 val) {
  return bfloat162_to_fp8x2_e4m3(val);
}
#endif // ENABLE_FP8

} // namespace eps
