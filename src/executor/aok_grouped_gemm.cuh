#pragma once

#include "src/scheduler/gemm_executor_interface.cuh"

#include "src/gemm/aok_grouped_gemm/include/aok/operations/default_hopper_grouped_gemm/default_hopper_gmm_kernel.h"

namespace eps {

template <typename InputType, typename OutputType, typename GroupType>
class AokGroupedGemm : public GemmInterface {
 public:
  using RunParams = typename GemmInterface::RunParams;

  struct Params {
    int num_local_experts;
    int sm_count;
  };

  AokGroupedGemm(Params p);

  void run(RunParams rp, cudaStream_t stream) override;

  size_t getWorkspaceSize(int64_t cols, int64_t max_num_tokens, int64_t local_num_experts) override;

  size_t output_type_bytes() override { return sizeof(OutputType); }

  Params p_;
 private:
  aok::AokFwdDefaultHopperGmm<InputType, OutputType, GroupType> aok_grouped_gemm_;
  size_t ws_size_{};
};

}  // namespace eps