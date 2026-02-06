#pragma once

#include <memory>
#include <cuda_runtime.h>

namespace aok {

/* 
  InputType/OutputType设置为bf16/half
  GroupType表明total_rows_before_expert的数据类型
 */
template <typename InputType, typename OutputType, typename GroupType>
class AokFwdDefaultHopperGmm {
public:
  AokFwdDefaultHopperGmm();
  ~AokFwdDefaultHopperGmm();

  void run(const InputType        *A,
           const InputType        *B,
           OutputType             *C,
           const GroupType          *total_rows_before_expert,
           int64_t                total_rows,
           int64_t                gemm_n,
           int64_t                gemm_k,
           int                    num_experts,
           void                   *ws,
           size_t                 workspace_size,
           int                    sm_count,
           cudaStream_t           stream);
  
  void runV2(const InputType        *A,
           const InputType        *B,
           OutputType             *C,
           const GroupType        *total_rows_before_expert,
           int64_t                total_rows,
           int64_t                gemm_m, // deepep fix buffer m
           int64_t                gemm_n,
           int64_t                gemm_k,
           int                    num_experts,
           void                   *ws,
           size_t                 workspace_size,
           int                    sm_count,
           cudaStream_t           stream);
  
  size_t getWorkSpace(int64_t gemm_n,
                      int64_t gemm_k,
                      int64_t total_rows,
                      int num_experts,
                      int sm_count);

private:
  class FwdGmmImpl;
  std::unique_ptr<FwdGmmImpl> kernel_impl_;
};

} // namespace aok