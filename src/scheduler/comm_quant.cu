#include "src/scheduler/comm_quant.cuh"
#include "src/quant/fp8.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace eps
{

    template <typename SrcType_, typename DstType_>
    const DstType_ *NonCommQuant<SrcType_, DstType_>::run(const SrcType *input, void *ws, cudaStream_t stream)
    {
        return input;
    }

    template <typename SrcType_, typename DstType_>
    const DstType_ *StaticCommQuant<SrcType_, DstType_>::run(const SrcType *input, void *ws, cudaStream_t stream)
    {
        static_scaled_fp8_quant((DstType *)ws, input, p_.scale, p_.num_tokens * p_.hidden_size, p_.hidden_size, stream);
        return (DstType_ *)ws;
    }

    template <typename SrcType_, typename DstType_>
    const DstType_ *DynamicCommQuant<SrcType_, DstType_>::run(const SrcType *input, void *ws, cudaStream_t stream)
    {

        /*
        AllReduceMaxScaleKernelParams p{
            .smChans = p_.smChans,
            .scale = p_.scale,
            .gathered_scales = p_.gathered_scales,
            .my_rank = p_.rank,
            .ep_world_size = p_.ep_world_size};

        dynamic_scaled_fp8_quant((DstType *)ws, input, p, p_.num_tokens * p_.hidden_size, p_.hidden_size, stream);
        */
        return (DstType_ *)ws;
    }

template
class NonCommQuant<float, float>;

template
class StaticCommQuant<float, __nv_fp8_e4m3>;

template
class DynamicCommQuant<float, __nv_fp8_e4m3>;

template
class NonCommQuant<half, half>;

template
class StaticCommQuant<half, __nv_fp8_e4m3>;

template
class DynamicCommQuant<half, __nv_fp8_e4m3>;

#ifdef ENABLE_BF16
template
class NonCommQuant<__nv_bfloat16, __nv_bfloat16>;

template
class StaticCommQuant<__nv_bfloat16, __nv_fp8_e4m3>;

template
class DynamicCommQuant<__nv_bfloat16, __nv_fp8_e4m3>;
#endif

}
