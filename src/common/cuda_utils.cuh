#pragma once

#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <type_traits>

#include "src/common/cuda_types.cuh"

namespace eps {

template <typename T> struct PackedType;
template <> struct PackedType<float> {
  using type = float;
};
template <> struct PackedType<int> {
  using type = int;
};
template <> struct PackedType<half> {
  using type = half2;
};
template <> struct PackedType<__nv_fp8_e4m3> {
  using type = __nv_fp8x2_e4m3;
};
#ifdef ENABLE_BF16
template <> struct PackedType<__nv_bfloat16> {
  using type = __nv_bfloat162;
};
#endif

template <typename T> struct NumElems;
template <> struct NumElems<float> {
  static constexpr int value = 1;
};
template <> struct NumElems<int> {
  static constexpr int value = 1;
};
template <> struct NumElems<half2> {
  static constexpr int value = 2;
};
template <> struct NumElems<__nv_fp8x2_e4m3> {
  static constexpr int value = 2;
};
#ifdef ENABLE_BF16
template <> struct NumElems<__nv_bfloat162> {
  static constexpr int value = 2;
};
#endif

/*
template <typename T_OUT, typename T_IN>
__host__ __device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <> __device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <> __device__ inline float2 cuda_cast<float2, half2>(half2 val) {
  return __half22float2(val);
}

template <> __device__ inline half2 cuda_cast<half2, float2>(float2 val) {
  return __float22half2_rn(val);
}
*/

template <typename T> struct SiluActivation {
  using return_type = float;
  static __device__ __forceinline__ T apply(const T &val) {
    return (T)((float)val / (1.0f + __expf((float)-val)));
  }
};
template <> struct SiluActivation<half2> {
  using return_type = float2;
  static __device__ __forceinline__ float2 apply(const half2 &val) {
    return float2{SiluActivation<float>::apply(val.x),
                  SiluActivation<float>::apply(val.y)};
  }
};
template <> struct SiluActivation<__nv_bfloat162> {
  using return_type = float2;
  static __device__ __forceinline__ float2 apply(const __nv_bfloat162 &val) {
    return float2{SiluActivation<float>::apply(val.x),
                  SiluActivation<float>::apply(val.y)};
  }
};

inline __device__ float2 operator+(float2 a, float b) {
  return make_float2(a.x + b, a.y + b);
}

inline __device__ float2 operator-(float2 a, float b) {
  return make_float2(a.x - b, a.y - b);
}

inline __device__ float2 operator*(float2 a, float b) {
  return make_float2(a.x * b, a.y * b);
}

inline __device__ float2 operator/(float2 a, float b) {
  return make_float2(a.x / b, a.y / b);
}

inline __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __device__ float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float2 operator*(float2 a, float2 b) {
  return float2{a.x * b.x, a.y * b.y};
}

inline __device__ float2 operator/(float2 a, float2 b) {
  return make_float2(a.x / b.x, a.y / b.y);
}

inline __device__ __nv_bfloat162 operator*(__nv_bfloat162 a, __nv_bfloat16 b)
{
  return __nv_bfloat162{a.x * b, a.y * b};
}

inline __device__ __half2 operator*(__half2 a, __half b) 
{
    return __half2{a.x * b, a.y * b};
}

template <typename T>
struct __align__(16) VectorizedType
{
  static_assert(sizeof(int4) % sizeof(T) == 0);
  static_assert(sizeof(int4) % sizeof(PackedType<T>) == 0);

  constexpr static int ItemsPerVec = sizeof(int4) / sizeof(T);
  constexpr static int PackPerVec = sizeof(int4) / sizeof(PackedType<T>::type);

  int4 data{};

  inline __device__ float absmax()
  {
    T *ptr = (T *)&data;
    float absmax_val = 0.0f;

    #pragma unroll
    for (int i = 0; i < ItemsPerVec; i++)
    {
      // absmax_val = max(absmax_val, fabs(ptr[i]));
      float val = static_cast<float>(ptr[i]);
      absmax_val = std::max(absmax_val, fabs(val));
    }

    return absmax_val;
  }

  inline __device__ T operator()(int i) {
    return *((T *)&data + i);
  }
};

template <typename T, typename U>
inline __device__ VectorizedType<T> operator*(VectorizedType<T> a, U b)
{
  VectorizedType<T> c;

  using PackedType = typename PackedType<T>::type;
  auto a_ptr = (PackedType *)(&a);
  auto c_ptr = (PackedType *)(&c);
#pragma unroll
  for (int i = 0; i < VectorizedType<T>::PackPerVec; i++)
  {
    c_ptr[i] = a_ptr[i] * b;
  }
  return c;
}

template <typename T>
inline __device__ VectorizedType<T> operator+(VectorizedType<T> a, VectorizedType<T> b)
{
  VectorizedType<T> c;

  using PackedType = typename PackedType<T>::type;
  auto a_ptr = (PackedType *)(&a);
  auto b_ptr = (PackedType *)(&b);
  auto c_ptr = (PackedType *)(&c);

#pragma unroll
  for (int i = 0; i < VectorizedType<T>::PackPerVec; i++)
  {
    c_ptr[i] = a_ptr[i] + b_ptr[i];
  }
  return c;
}

template <typename T>
inline __device__ VectorizedType<T>& operator+=(VectorizedType<T>& a, const VectorizedType<T>& b)
{
  using PackedType = typename PackedType<T>::type;
  auto a_ptr = (PackedType *)(&a);
  auto b_ptr = (PackedType *)(&b);

#pragma unroll
  for (int i = 0; i < VectorizedType<T>::PackPerVec; i++)
  {
    a_ptr[i] += b_ptr[i];
  }

  return a;
}

template <typename T, int N>
struct Array {
  using value_type = T;
  using size_type = int;
  using difference_type = int;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  static_assert(N > 0);

  T __a[N];

  __device__ __host__ constexpr reference operator[](size_type i) {
    return __a[i];
  }
  __device__ __host__ constexpr const_reference operator[](size_type i) const {
    return __a[i];
  }

  __device__ __host__ constexpr reference front() {
    return *begin();
  }

  __device__ __host__ constexpr const_reference front() const {
    return *begin();
  }

  __device__ __host__ constexpr reference back() {
    return *(end() - 1);
  }

  __device__ __host__ constexpr const_reference back() const {
    return *(end() - 1);
  }

  __device__ __host__ constexpr pointer data() {
    return &__a[0];
  }

  __device__ __host__ constexpr const_pointer data() const {
    return &__a[0];
  }

  __device__ __host__ constexpr iterator begin() {
    return data();
  }

  __device__ __host__ constexpr const_iterator begin() const {
    return data();
  }

  __device__ __host__ constexpr iterator end() {
    return data() + N;
  }

  __device__ __host__ constexpr const_iterator end() const {
    return data() + N;
  }

  __device__ __host__ constexpr std::integral_constant<int, N> size() const {
    return {};
  }

  __device__ __host__ constexpr std::false_type empty() const {
    return {};
  }
};

struct uint4_t {
};

template<class T>
struct bitsof_t: std::integral_constant<int, sizeof(T) * 8> {
};

template<>
struct bitsof_t<uint4_t>: std::integral_constant<int, 4> {
};

template<class T>
inline constexpr bitsof_t<T> bitsof{};

namespace detail {

struct __uint4_t {
    uint32_t x;
};

}  // namespace detail

template<class T>
struct SubBytePtr {

    __device__ T& operator[](int i)
    {
        return *reinterpret_cast<T*>(ptr_ + i * bitsof<T> / bitsof<char>);
    }

    friend __device__ SubBytePtr operator+(const SubBytePtr a, int n)
    {
        return SubBytePtr{a.ptr_ + n * bitsof<T> / bitsof<char>};
    }

    friend __device__ SubBytePtr operator+(int n, const SubBytePtr a)
    {
        return a + n;
    }

    __device__ explicit operator T*() const
    {
        return (T*)ptr_;
    }

    char* ptr_;
};

template<class T, class SFINAE = void>
struct get_pointer_type_t {
    using type = T*;
};

template<class T>
struct get_pointer_type_t<T, std::enable_if_t<bitsof<T> % 8 != 0>> {
    using type = SubBytePtr<T>;
};

template<class T>
using get_pointer_type = typename get_pointer_type_t<T>::type;

template<int N>
struct Array<uint4_t, N> {
    using value_type      = detail::__uint4_t;
    using size_type       = int;
    using difference_type = int;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = SubBytePtr<uint4_t>;
    using const_pointer   = SubBytePtr<const uint4_t>;

    static_assert(N % 8 == 0);

    detail::__uint4_t __a[N / 8];

    __device__ __host__ constexpr reference operator[](size_type i) noexcept
    {
        return __a[i / 8];
    }
    __device__ __host__ constexpr const_reference operator[](size_type i) const noexcept
    {
        return __a[i / 8];
    }

    __device__ __host__ constexpr std::integral_constant<int, N> size() const noexcept
    {
        return {};
    }

    __device__ __host__ constexpr std::false_type empty() const noexcept
    {
        return {};
    }

    __device__ __host__ constexpr pointer data() noexcept
    {
        return {(char*)&__a[0]};
    }
};

static_assert(sizeof(Array<uint4_t, 8>) == 4);
static_assert(sizeof(Array<uint4_t, 16>) == 8);
static_assert(sizeof(Array<uint4_t, 24>) == 12);
static_assert(sizeof(Array<uint4_t, 32>) == 16);

template <int... Ns>
struct Shape {
  static constexpr Array<int, sizeof...(Ns)> data_{Ns...};

  constexpr Shape() = default;

  Shape(std::integral_constant<int, Ns>...) {}

  template <int index>
  constexpr auto get() const {
    return std::integral_constant<int, data_[index]>{};
  }

  constexpr auto m() const {
    return get<0>();
  }

  constexpr auto n() const {
    return get<1>();
  }

  constexpr auto k() const {
    return get<2>();
  }

  constexpr int c() const {
    return get<0>();
  }

  constexpr int s() const {
    return get<1>();
  }

  constexpr int count() const {
    return (Ns * ...);
  }
};

template<class T>
__device__ constexpr T Max(T a, T b)
{
    if constexpr (std::is_same_v<T, half>) {
        return __hmax(a, b);
    }

#ifdef ENABLE_BF16
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __hmax(a, b);
    }
#endif
#endif

    if constexpr (std::is_same_v<T, float>) {
        return fmaxf(a, b);
    }

    if constexpr (std::is_same_v<T, int>) {
        return max(a, b);
    }

    return T{};
}

template<class T>
__device__ constexpr T Min(T a, T b)
{
    if constexpr (std::is_same_v<T, half>) {
        return __hmin(a, b);
    }

#ifdef ENABLE_BF16
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __hmin(a, b);
    }
#endif
#endif

    if constexpr (std::is_same_v<T, float>) {
        return fminf(a, b);
    }

    if constexpr (std::is_same_v<T, int>) {
        return min(a, b);
    }

    return T{};
}

template<int WarpThreadC, class T, int C>
__device__ inline void warp_minmax(Array<T, 2>& stats, const Array<T, C>& x)
{
  #pragma unroll
  for (int i = 0; i < C; ++i) {
    stats[0] = Min(stats[0], x[i]);
    stats[1] = Max(stats[1], x[i]);
  }
  if constexpr (sizeof(T) == 2) {
    #pragma unroll
    for (int mask = WarpThreadC / 2; mask > 0; mask /= 2) {
      Array<T, 2> tmp;
      (uint32_t&)tmp = __shfl_xor_sync(uint32_t(-1), (uint32_t&)stats, mask);
      stats[0]       = Min(stats[0], tmp[0]);
      stats[1]       = Max(stats[1], tmp[1]);
    }
  }
  else {
    #pragma unroll
    for (int mask = WarpThreadC / 2; mask > 0; mask /= 2) {
      stats[0] = Min(stats[0], __shfl_xor_sync(uint32_t(-1), stats[0], mask));
      stats[1] = Max(stats[1], __shfl_xor_sync(uint32_t(-1), stats[1], mask));
    }
  }
}

template<typename T, typename Tq>
__device__ inline Array<T, 2> get_scale(Array<T, 2>& stats) {
  static constexpr int BitNum = bitsof_t<Tq>{};
  static constexpr int MaxVal = (1UL << BitNum) - 1;

  Array<T, 2> sz{};
  sz[0] = (stats[1] - stats[0]) / cuda_cast<T>(MaxVal);
  sz[1] = stats[0];

  return sz;
}

template <typename T, int N>
inline __device__ void Store(T* dst, const Array<T, N>& src) {
  static constexpr int U4_SIZE = sizeof(uint4);
  static constexpr int VEC_SIZE = sizeof(Array<T, N>);
  static constexpr int U4_NUM = VEC_SIZE / U4_SIZE;
  static constexpr int SIZE_TAIL = VEC_SIZE % U4_SIZE;

  if constexpr (VEC_SIZE <= U4_SIZE) {
    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
      *(uint4*)dst = (const uint4&)src;
    } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
      *(uint2*)dst = (const uint2&)src;
    } else if constexpr (sizeof(Array<T, N>) == sizeof(uint1)) {
      *(uint1*)dst = (const uint1&)src;
    } else {
      static_assert(!std::is_same_v<T, T>);
    }
  } else if constexpr (SIZE_TAIL == 0) {
    uint4* dst_ = (uint4*)dst;
    const uint4* src_ = (const uint4*)(&src);
    #pragma unroll
    for (int i = 0; i < U4_NUM; ++i) {
      dst_[i] = src_[i];
    }
  } else {
    static_assert(!std::is_same_v<T, T>);
  }
}

template <typename T, int N>
inline __device__ void Ldg(Array<T, N>& dst, const T* src) {
  static constexpr int U4_SIZE = sizeof(uint4);
  static constexpr int VEC_SIZE = sizeof(Array<T, N>);
  static constexpr int U4_NUM = VEC_SIZE / U4_SIZE;
  static constexpr int SIZE_TAIL = VEC_SIZE % U4_SIZE;

  if constexpr (VEC_SIZE <= U4_SIZE) {
    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
      (uint4&)dst = __ldg((const uint4*)src);
    } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
      (uint2&)dst = __ldg((const uint2*)src);
    } else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
      (uint&)dst = __ldg((const uint*)src);
    } else {
      static_assert(!std::is_same_v<T, T>);
    }
  } else if constexpr (SIZE_TAIL == 0) {
    uint4* dst_ = (uint4*)(&dst);
    const uint4* src_ = (const uint4*)src;
    #pragma unroll
    for (int i = 0; i < U4_NUM; ++i) {
      dst_[i] = __ldg(src_ + i);
    }
  } else {
    static_assert(!std::is_same_v<T, T>);
  }
}

}
