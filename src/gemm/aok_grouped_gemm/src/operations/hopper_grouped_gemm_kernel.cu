#include "hopper_grouped_gemm.h"
#include "aok/operations/default_hopper_grouped_gemm/default_hopper_gmm_kernel.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace aok {


template <typename T>
using CutlassType = cute::conditional_t<cute::is_same_v<T, __nv_bfloat16>, cutlass::bfloat16_t, cutlass::half_t>;

template <typename InputType, typename OutputType, typename GroupType>
class AokFwdDefaultHopperGmm<InputType, OutputType, GroupType>::FwdGmmImpl {

using CutlassInputType = CutlassType<InputType>;
using CutlassOutputType = CutlassType<OutputType>;

public:
  FwdGmmImpl() {
    kernel_runner_ = std::make_shared<HopperGmm<CutlassInputType, CutlassOutputType, GroupType>>();
  }

  void forward(const InputType           *A,
              const InputType            *B,
              OutputType                  *C,
              const GroupType      *total_rows_before_expert,
              int64_t             total_rows,
              int64_t             gemm_n,
              int64_t             gemm_k,
              int                 num_experts,
              void               *ws,
              size_t              workspace_size,
              int                 sm_count,
              cudaStream_t        st)
  {
    kernel_runner_->Run((const CutlassInputType*)A,
                        (const CutlassInputType*)B,
                        (CutlassOutputType *)C,
                        total_rows_before_expert,
                        total_rows,
                        gemm_n,
                        gemm_k,
                        num_experts,
                        ws,
                        workspace_size,
                        sm_count,
                        st);
  }

  void forwardV2(const InputType           *A,
              const InputType            *B,
              OutputType                  *C,
              const GroupType      *total_rows_before_expert,
              int64_t             total_rows,
              int64_t             gemm_m,
              int64_t             gemm_n,
              int64_t             gemm_k,
              int                 num_experts,
              void               *ws,
              size_t              workspace_size,
              int                 sm_count,
              cudaStream_t        st)
  {
    kernel_runner_->RunV2((const CutlassInputType*)A,
                        (const CutlassInputType*)B,
                        (CutlassOutputType *)C,
                        total_rows_before_expert,
                        total_rows,
                        gemm_m,
                        gemm_n,
                        gemm_k,
                        num_experts,
                        ws,
                        workspace_size,
                        sm_count,
                        st);
  }

  size_t getWorkSpace(int64_t gemm_n,
                      int64_t gemm_k,
                      int64_t total_rows,
                      int num_experts,
                      int sm_count)
  {
    return kernel_runner_->GetWorkSpace(gemm_n,
                                        gemm_k,
                                        total_rows,
                                        num_experts,
                                        sm_count);
  }

private:
  std::shared_ptr<HopperGmm<CutlassInputType, CutlassOutputType, GroupType>> kernel_runner_;
};

template <typename InputType, typename OutputType, typename GroupType>
AokFwdDefaultHopperGmm<InputType, OutputType, GroupType>::AokFwdDefaultHopperGmm() : kernel_impl_(std::make_unique<FwdGmmImpl>()) {
}

template <typename InputType, typename OutputType, typename GroupType>
AokFwdDefaultHopperGmm<InputType, OutputType, GroupType>::~AokFwdDefaultHopperGmm() = default;

template <typename InputType, typename OutputType, typename GroupType>
void AokFwdDefaultHopperGmm<InputType, OutputType, GroupType>::run(const InputType          *A,
                                  const InputType            *B,
                                  OutputType                  *C,
                                  const GroupType      *total_rows_before_expert,
                                  int64_t             total_rows,
                                  int64_t             gemm_n,
                                  int64_t             gemm_k,
                                  int                 num_experts,
                                  void                *ws,
                                  size_t              workspace_size,
                                  int                 sm_count,
                                  cudaStream_t        st) {
  kernel_impl_->forward(A,
                        B,
                        C,
                        total_rows_before_expert,
                        total_rows,
                        gemm_n,
                        gemm_k,
                        num_experts,
                        ws,
                        workspace_size,
                        sm_count,
                        st);        
}

template <typename InputType, typename OutputType, typename GroupType>
void AokFwdDefaultHopperGmm<InputType, OutputType, GroupType>::runV2(const InputType          *A,
                                    const InputType            *B,
                                    OutputType                  *C,
                                    const GroupType      *total_rows_before_expert,
                                    int64_t             total_rows,
                                    int64_t             gemm_m,
                                    int64_t             gemm_n,
                                    int64_t             gemm_k,
                                    int                 num_experts,
                                    void                *ws,
                                    size_t              workspace_size,
                                    int                 sm_count,
                                    cudaStream_t        st) {
  kernel_impl_->forwardV2(A,
                        B,
                        C,
                        total_rows_before_expert,
                        total_rows,
                        gemm_m,
                        gemm_n,
                        gemm_k,
                        num_experts,
                        ws,
                        workspace_size,
                        sm_count,
                        st);        
}

template <typename InputType, typename OutputType, typename GroupType>
size_t AokFwdDefaultHopperGmm<InputType, OutputType, GroupType>::getWorkSpace(int64_t gemm_n,
                                        int64_t gemm_k,
                                        int64_t total_rows,
                                        int num_experts,
                                        int sm_count)
{
  return kernel_impl_->getWorkSpace(gemm_n,
                                    gemm_k,
                                    total_rows,
                                    num_experts,
                                    sm_count);
}



template class AokFwdDefaultHopperGmm<__nv_bfloat16, __nv_bfloat16, int32_t>;

} // namespace aok