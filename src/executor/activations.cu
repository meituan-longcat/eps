#include "src/executor/activations.cuh"

#include "src/scheduler/utils.cuh"
#include "src/common/cuda_utils.cuh"

namespace eps
{
  template <typename T, int BLOCK_SIZE, int INTER_SIZE_X2>
  __global__ void silu_activate_kernel(const T *gate_up, T *result_, const int *exclusive_sum, int num_experts)
  {
    using VectorizedType = VectorizedType<T>;
    using PackedType = typename PackedType<T>::type;
    static_assert(INTER_SIZE_X2 / 2 % VectorizedType::ItemsPerVec == 0);

    int begin = exclusive_sum[0];
    int end = exclusive_sum[num_experts];
    for (int token_idx = blockIdx.x; token_idx < end - begin; token_idx += gridDim.x)
    {
      const T *gate = gate_up + (begin + token_idx) * INTER_SIZE_X2;
      const T *up = gate + INTER_SIZE_X2 / 2;
      T *result = result_ + (begin + token_idx) * INTER_SIZE_X2 / 2;

#pragma unroll
      for (int i = threadIdx.x; i < INTER_SIZE_X2 / 2 / VectorizedType::ItemsPerVec; i += blockDim.x)
      {
        auto gate_vec = reinterpret_cast<const VectorizedType *>(gate)[i];
        auto up_vec = reinterpret_cast<const VectorizedType *>(up)[i];
        auto& result_vec = reinterpret_cast<VectorizedType *>(result)[i];

#pragma unroll
        for (int j = 0; j < VectorizedType::PackPerVec; j++)
        {
          auto gate_pack = reinterpret_cast<const PackedType *>(&gate_vec)[j];
          auto up_pack = reinterpret_cast<const PackedType *>(&up_vec)[j];
          auto result_pack = SiluActivation<PackedType>::apply(gate_pack) * cuda_cast<SiluActivation<PackedType>::return_type>(up_pack);
          reinterpret_cast<PackedType *>(&result_vec)[j] = cuda_cast<PackedType>(result_pack);
        }
      }
    }
  }

  template <typename T, int BLOCK_SIZE, int INTER_SIZE_X2>
  __global__ void silu_activate_kernel(const T *gate_up, T *result_, const int num_tokens)
  {
    using VectorizedType = VectorizedType<T>;
    using PackedType = typename PackedType<T>::type;
    static_assert(INTER_SIZE_X2 / 2 % VectorizedType::ItemsPerVec == 0);

    for (int token_idx = blockIdx.x; token_idx < num_tokens; token_idx += gridDim.x)
    {
      const T *gate = gate_up + token_idx * INTER_SIZE_X2;
      const T *up = gate + INTER_SIZE_X2 / 2;
      T *result = result_ + token_idx * INTER_SIZE_X2 / 2;

#pragma unroll
      for (int i = threadIdx.x; i < INTER_SIZE_X2 / 2 / VectorizedType::ItemsPerVec; i += blockDim.x)
      {
        auto gate_vec = reinterpret_cast<const VectorizedType *>(gate)[i];
        auto up_vec = reinterpret_cast<const VectorizedType *>(up)[i];
        auto& result_vec = reinterpret_cast<VectorizedType *>(result)[i];

#pragma unroll
        for (int j = 0; j < VectorizedType::PackPerVec; j++)
        {
          auto gate_pack = reinterpret_cast<const PackedType *>(&gate_vec)[j];
          auto up_pack = reinterpret_cast<const PackedType *>(&up_vec)[j];
          auto result_pack = SiluActivation<PackedType>::apply(gate_pack) * cuda_cast<SiluActivation<PackedType>::return_type>(up_pack);
          reinterpret_cast<PackedType *>(&result_vec)[j] = cuda_cast<PackedType>(result_pack);
        }
      }
    }
  }

#define INTER_SIZE_X2_SWITCH(INTER_SIZE_X2_, ...)                                                  \
  [&] {                                                                                            \
    if (INTER_SIZE_X2_ == 256 * 2)                                                                 \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 256 * 2;                                                \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 512 * 2)                                                            \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 512 * 2;                                                \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 768 * 2)                                                            \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 768 * 2;                                                \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 1024 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 1024 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 1408 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 1408 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 1536 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 1536 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 2048 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 2048 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 2880 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 2880 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 2560 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 2560 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 5504 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 5504 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 6912 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 6912 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 3584 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 3584 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 8192 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 8192 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 9728 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 9728 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 14336 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 14336 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                               \
    else if (INTER_SIZE_X2_ == 17408 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 17408 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                               \
    else if (INTER_SIZE_X2_ == 20480 * 2)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 20480 * 2;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else if (INTER_SIZE_X2_ == 36864)                                                           \
    {                                                                                              \
      constexpr static int INTER_SIZE_X2 = 36864;                                               \
      return __VA_ARGS__();                                                                        \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
      throw std::runtime_error("Not supported inter_size_x2_: " + std::to_string(INTER_SIZE_X2_)); \
    }                                                                                              \
  }

  template <typename T>
  void silu_activate(const T *gate_up, T *result_, const int *exclusive_sum, int num_experts, int inter_size_x2, int num_tokens_hint, cudaStream_t stream)
  {
    int num_blocks;
    int block_size;
    auto num_blocks_heuristic = [&]()
    {
      num_blocks = num_tokens_hint;
      if (inter_size_x2 >= 1024 * 2)
      {
        num_blocks = std::min(num_blocks, 1024);
        block_size = 512;
        if (num_blocks > 256)
        {
          block_size = 256;
        }
      }
      else
      {
        num_blocks = std::min(num_blocks, 2048);
        block_size = 64;
      }
    };
    num_blocks_heuristic();

    INTER_SIZE_X2_SWITCH(inter_size_x2,
      BLOCK_SIZE_SWITCH(block_size,
                        [&]() {
                          silu_activate_kernel<T, BLOCK_SIZE, INTER_SIZE_X2><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                            gate_up, result_, exclusive_sum, num_experts);
                            }
      )
    )();
  }

  template <typename T>
  void silu_activate(const T *gate_up, T *result_, int inter_size_x2, int num_tokens, cudaStream_t stream)
  {
    int num_blocks;
    int block_size;
    auto num_blocks_heuristic = [&]()
    {
      num_blocks = num_tokens;
      if (inter_size_x2 >= 1024 * 2)
      {
        num_blocks = std::min(num_blocks, 1024);
        block_size = 512;
        if (num_blocks > 256)
        {
          block_size = 256;
        }
      }
      else
      {
        num_blocks = std::min(num_blocks, 2048);
        block_size = 64;
      }
    };
    num_blocks_heuristic();

    INTER_SIZE_X2_SWITCH(inter_size_x2,
      BLOCK_SIZE_SWITCH(block_size,
                        [&]() {
                          silu_activate_kernel<T, BLOCK_SIZE, INTER_SIZE_X2><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                            gate_up, result_, num_tokens);
                            }
      )
    )();
  }

  template void silu_activate(const half *gate_up, half *result_, const int *exclusive_sum, int num_experts, int inter_size_x2, int num_tokens_hint, cudaStream_t stream);
  template void silu_activate(const float *gate_up, float *result_,
                              const int *exclusive_sum, int num_experts,
                              int inter_size_x2, int num_tokens_hint,
                              cudaStream_t stream);
#ifdef ENABLE_BF16
  template void silu_activate(const __nv_bfloat16 *gate_up,
                              __nv_bfloat16 *result_, const int *exclusive_sum,
                              int num_experts, int inter_size_x2,
                              int num_tokens_hint, cudaStream_t stream);
#endif

template<typename T>
void Activation<T>::run(RunParams p, cudaStream_t stream) {
  silu_activate((T*)p.gate_up,
                (T*)p.activation,
                p.exclusive_sum,
                p.num_experts_per_stage,
                p.inter_size_x2,
                p.num_tokens_hint,
                stream);
}

template<typename T>
void DenseActivation<T>::run(RunParams p, cudaStream_t stream) {
  silu_activate((T*)p.gate_up,
                (T*)p.activation,
                p.inter_size_x2,
                p.num_tokens,
                stream);
}

template
class Activation<float>;

template
class Activation<half>;

#ifdef ENABLE_BF16
template
class Activation<__nv_bfloat16>;
#endif

template
class DenseActivation<float>;

template
class DenseActivation<half>;

#ifdef ENABLE_BF16
template
class DenseActivation<__nv_bfloat16>;
#endif

#undef INTER_SIZE_X2_SWITCH

template <typename T, typename VectorizedType = int4>
__global__ __launch_bounds__(1024) void activation_kernel(
    T *full_gate, const T *full_up, const int *exclusive_sum,
    size_t num_experts_per_stage, size_t n) {
  
  using PackedType = typename PackedType<T>::type;
  
  size_t begin_row = exclusive_sum[0];
  size_t end_row = exclusive_sum[num_experts_per_stage];
  size_t m = end_row - begin_row;

  PackedType *gate = reinterpret_cast<PackedType*>(full_gate + begin_row * n);
  const PackedType *up = reinterpret_cast<const PackedType*>(full_up + begin_row * n);

  size_t num_elems_of_packed_type = NumElems<PackedType>::value;
  size_t num_elems = m * n / num_elems_of_packed_type;
  assert(m * n % num_elems_of_packed_type == 0);

  size_t packed_elems_per_vec = sizeof(VectorizedType) / sizeof(PackedType);
  size_t vec_elems = num_elems / packed_elems_per_vec;

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t offset = gridDim.x * blockDim.x;

#pragma unroll
  for (int i = tid; i < vec_elems; i += offset) {
    VectorizedType gate_vec_elem = reinterpret_cast<VectorizedType *>(gate)[i];
    const VectorizedType up_vec_elem = reinterpret_cast<const VectorizedType *>(up)[i];

    for (int j = 0; j < packed_elems_per_vec; j++) {
      reinterpret_cast<PackedType *>(&gate_vec_elem)[j] = cuda_cast<PackedType>(
          SiluActivation<PackedType>::apply(
              reinterpret_cast<PackedType *>(&gate_vec_elem)[j]) *
          cuda_cast<SiluActivation<PackedType>::return_type>(
              reinterpret_cast<const PackedType *>(&up_vec_elem)[j]));
    }

    reinterpret_cast<VectorizedType *>(gate)[i] = gate_vec_elem;
  }

  if (tid == 0) {
#pragma unroll
    for (int i = vec_elems * packed_elems_per_vec; i < num_elems; i++) {
      gate[i] = cuda_cast<PackedType>(
          SiluActivation<PackedType>::apply(gate[i]) *
          cuda_cast<SiluActivation<PackedType>::return_type>(up[i]));
    }
  }
}

template <typename T, typename VectorizedType = int4>
__global__ __launch_bounds__(1024) void activation_kernel2(
    T *full_activation, const T *full_gate_up, const int *exclusive_sum,
    size_t num_experts_per_stage, size_t n2) {

  using PackedType = typename PackedType<T>::type;

  size_t num_elems_of_packed_type = NumElems<PackedType>::value;
  size_t n = n2 / 2;
  size_t num_elems_per_row = n / num_elems_of_packed_type;
  assert(n % num_elems_of_packed_type == 0);
  
  size_t begin_row = exclusive_sum[0];
  size_t end_row = exclusive_sum[num_experts_per_stage];
  size_t m = end_row - begin_row;

  PackedType *activation = reinterpret_cast<PackedType*>(full_activation + begin_row * n);
  const PackedType *gate_up = reinterpret_cast<const PackedType*>(full_gate_up + begin_row * n2);

  size_t packed_elems_per_vec = sizeof(VectorizedType) / sizeof(PackedType);
  size_t vec_elems_per_row = num_elems_per_row / packed_elems_per_vec;

  size_t tid = threadIdx.x;
  size_t row_id = blockIdx.x;

  for (int i = row_id; i < m; i += gridDim.x) {
    for (int j = tid; j < vec_elems_per_row; j += blockDim.x) {
      VectorizedType gate_vec_elem = reinterpret_cast<const VectorizedType *>(
          gate_up + i * num_elems_per_row * 2)[j];
      VectorizedType up_vec_elem = reinterpret_cast<const VectorizedType *>(
          gate_up + i * num_elems_per_row * 2 + num_elems_per_row)[j];

      for (int jj = 0; jj < packed_elems_per_vec; jj++) {
        reinterpret_cast<PackedType *>(&gate_vec_elem)[jj] =
            cuda_cast<PackedType>(
                SiluActivation<PackedType>::apply(
                    reinterpret_cast<PackedType *>(&gate_vec_elem)[jj]) *
                cuda_cast<SiluActivation<PackedType>::return_type>(
                    reinterpret_cast<PackedType *>(&up_vec_elem)[jj]));
      }

      reinterpret_cast<VectorizedType *>(
          activation + i * num_elems_per_row)[j] = gate_vec_elem;
    }

    for (int jj = vec_elems_per_row * packed_elems_per_vec;
         jj < num_elems_per_row; jj++) {
      activation[i * num_elems_per_row + jj] = cuda_cast<PackedType>(
          SiluActivation<PackedType>::apply(
              gate_up[i * num_elems_per_row * 2 + jj]) *
          cuda_cast<SiluActivation<PackedType>::return_type>(
              gate_up[i * num_elems_per_row * 2 + num_elems_per_row + jj]));
    }
  }
}

template <typename T>
void activationLauncher(T *gate, T *up, const int *exclusive_sum,
                        size_t num_experts_per_stage, size_t n,
                        cudaStream_t stream) {
  dim3 block(1024);
  dim3 grid(108); // TODO: more accurate

  activation_kernel<T><<<grid, block, 0, stream>>>
      (gate, up, exclusive_sum, num_experts_per_stage, n);
}

template <typename T>
void activationLauncher2(T *activation, T *gate_up,
                         const int *exclusive_sum, size_t num_experts_per_stage,
                         size_t n2, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(108); // TODO: more accurate

  activation_kernel2<T><<<grid, block, 0, stream>>>
      (activation, gate_up, exclusive_sum, num_experts_per_stage, n2);
}

template<typename T>
void ActivationUnfused<T>::run(RunParams p, cudaStream_t stream) {
  activationLauncher(
    (T*)p.gate,
    (T*)p.up,
    p.exclusive_sum,
    p.num_experts_per_stage,
    p.inter_size,
    stream
  );
}

template
class ActivationUnfused<half>;

#ifdef ENABLE_BF16
template
class ActivationUnfused<__nv_bfloat16>;
#endif

template
void activationLauncher(half *gate, half *up, const int *exclusive_sum,
                        size_t num_experts_per_stage, size_t n,
                        cudaStream_t stream);

template
void activationLauncher(__nv_bfloat16 *gate, __nv_bfloat16 *up, const int *exclusive_sum,
                        size_t num_experts_per_stage, size_t n,
                        cudaStream_t stream);

template
void activationLauncher2(half *activation, half *gate_up,
                         const int *exclusive_sum, size_t num_experts_per_stage,
                         size_t n2, cudaStream_t stream);

template
void activationLauncher2(__nv_bfloat16 *activation, __nv_bfloat16 *gate_up,
                         const int *exclusive_sum, size_t num_experts_per_stage,
                         size_t n2, cudaStream_t stream);
}
