#include <cuda_fp8.h>

#include <iostream>

#include "src/common/debug.cuh"
#include "src/executor/aok_grouped_gemm.cuh"

namespace eps {

template <typename InputType, typename OutputType, typename GroupType>
AokGroupedGemm<InputType, OutputType, GroupType>::AokGroupedGemm(Params p) : p_{p} {
  ws_size_ = aok_grouped_gemm_.getWorkSpace(1, 1, 1, p_.num_local_experts, p_.sm_count);
}

template <typename InputType, typename OutputType, typename GroupType>
size_t AokGroupedGemm<InputType, OutputType, GroupType>::getWorkspaceSize(int64_t cols,
                                                                          int64_t max_num_tokens,
                                                                          int64_t local_num_experts) {
  return ws_size_;
}

template <typename InputType, typename OutputType, typename GroupType>
void AokGroupedGemm<InputType, OutputType, GroupType>::run(RunParams rp, cudaStream_t stream) {
  aok_grouped_gemm_.run((InputType *)rp.A,
                        (InputType *)rp.B,
                        (OutputType *)rp.C,
                        (GroupType *)rp.total_rows_before_expert,
                        rp.total_rows,
                        rp.gemm_n,
                        rp.gemm_k,
                        rp.num_experts_per_stage,
                        (void *)(rp.ws),
                        ws_size_,
                        p_.sm_count,
                        stream);
  sync_check_cuda_error_eps();
}

template class AokGroupedGemm<__nv_bfloat16, __nv_bfloat16, int>;

}  // namespace eps
