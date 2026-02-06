#include <cuda_fp8.h>

#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>

namespace eps {

using FP8_TYPE = __nv_fp8_e4m3;

template <typename scalar_t>
void static_scaled_fp8_quant(FP8_TYPE *out,         // [..., d]
                             const scalar_t *input, // [..., d]
                             const float *scale,    // [1]
                             size_t num_elems,
                             size_t num_cols,
                             cudaStream_t stream);

template <typename scalar_t>
void dynamic_scaled_fp8_quant(FP8_TYPE *out,         // [..., d]
                              const scalar_t *input, // [..., d]
                              float *scale,          // [1]
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream);

struct AllReduceMaxScaleKernelParams
{
  mscclpp::DeviceHandle<mscclpp::SmChannel> *smChans;
  float *scale;           /*[1]*/
  float *gathered_scales; /*[ep_world_size]*/
  int my_rank;
  int ep_world_size;
};

template <typename scalar_t>
void dynamic_scaled_fp8_quant(FP8_TYPE *out,         // [..., d]
                              const scalar_t *input, // [..., d]
                              AllReduceMaxScaleKernelParams p,
                              size_t num_elems,
                              size_t num_cols,
                              cudaStream_t stream);

template<typename T>
struct PerTokenGroupQuantFP8Params {
    FP8_TYPE* quanted;
    T* origin;
    float* scales;
    const int* exclusive_sum;
    int num_experts;
    int64_t cols;
    int group_size;
    int num_tokens_hint;
    bool dequant{};

    int num_thread_blocks() { return num_tokens_hint; }
};

template<typename T>
struct PerTokenGroupQuantFP8DenseParams {
    FP8_TYPE* quanted;
    T* origin;
    float* scales;
    int64_t cols;
    int group_size;
    int64_t num_tokens;
    bool dequant{};

    int num_thread_blocks() { return num_tokens; }
};

template <typename T, template<typename> typename Params>
void per_token_group_quant_fp8(Params<T> params, cudaStream_t stream);

}