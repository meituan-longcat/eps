#include "cutlass/cutlass.h"
#include "hopper_grouped_gemm.h"
#include "hopper_grouped_gemm_kernel.h"
#include <vector>
#include <memory>

namespace aok {

using namespace cute;
using cutlass::layout::RowMajor;
using cutlass::layout::ColumnMajor;


template <typename InputType, typename OutputType, typename GroupType>
struct HopperGmm<InputType, OutputType, GroupType>::Impl {

  struct pairHash {
    std::size_t operator()(const std::pair<int, int>& pair) const
    {
        return std::hash<int>()(pair.first) ^ std::hash<int>()(pair.second);
    }

    std::size_t operator()(const std::pair<int, std::pair<int, int>> &pair) const
    {
        return std::hash<int>()(pair.first) ^ std::hash<int>()(pair.second.first) ^ std::hash<int>()(pair.second.second);
    }
  };

  struct pairEqual {
      bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) const
      {
          return a.first == b.first && a.second == b.second;
      }
      bool operator()(const std::pair<int, std::pair<int, int>> &a, const std::pair<int, std::pair<int, int>> &b) const
      {
          return a.first == b.first && a.second.first == b.second.first && a.second.second == b.second.second;
      }
  };

  using Kernel = std::unique_ptr<Base_SM90_GROUPED_GEMM<InputType, OutputType, GroupType>>;
  // H20 default kernel,
  using SpecialKernels = std::unordered_map<std::pair<int, std::pair<int, int>>, std::unique_ptr<Base_SM90_GROUPED_GEMM<InputType, OutputType, GroupType>>, pairHash, pairEqual>;
  // 当前H20 disptach到default kernel时，bs已经很大了，此时走no swap
  using DefaultKernel = SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_16,_64>, Shape<_1,_1,_1>, _4, GroupType, true, false>;
  using DefaultKernelV2 = SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_16,_64>, Shape<_1,_1,_1>, _4, GroupType, false, false>;
  // H800 default kernel, H800算力很大, no swap ab,
  using H800DefaultKernel = SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_16,_64>, Shape<_1,_1,_1>, _4, GroupType, true, false>;
  using H800DefaultKernelV2 = SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_16,_64>, Shape<_1,_1,_1>, _4, GroupType, false, false>;

  using TileNKernels = std::unordered_map<int, std::unique_ptr<Base_SM90_GROUPED_GEMM<InputType, OutputType, GroupType>>>;

  template <bool IsPrefixSum=true, bool IsSwapAB=false>
  void Generate(SpecialKernels &special_kernels, TileNKernels &tile_n_kernels, Kernel &kernel)
  { 
    // H20 config
    // ep,  38/48B gate+up
    // gate & up
    tile_n_kernels.insert(
      std::make_pair(16,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_16,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 32,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_32,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 48,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_48,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 64,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_64,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 80,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_80,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 96,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_96,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(112,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_112,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(128,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_128,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(144,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_144,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(160,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_80,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(176,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_176,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));

    // down
    tile_n_kernels.insert(
     std::make_pair( 16,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_16,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 32,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_32,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 48,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_48,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 64,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_64,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 80,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_80,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
     std::make_pair( 96,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_96,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(112,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_112,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(128,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_128,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(144,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_144,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(160,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_80,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
    tile_n_kernels.insert(
      std::make_pair(176,
      new SM90_GROUPED_GEMM<RowMajor, ColumnMajor, RowMajor, InputType, OutputType, float, Shape<_128,_176,_64>, Shape<_1,_1,_1>, _5, GroupType, IsPrefixSum, IsSwapAB>{}));
  }

  void Run(const InputType       *A,
          const InputType        *B,
          OutputType              *C,
          const GroupType  *total_rows_before_expert,
          int            total_rows,
          int64_t        gemm_n,
          int64_t        gemm_k,
          int            num_experts,
          void           *ws,
          size_t         workspace_size,
          int            sm_count,
          cudaStream_t   st)
  {
    if(sm_count > 78) {
      // no h20
      // 默认算子, no swap ab
      h800_kernel_->Launch(A, B, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, ws, workspace_size, sm_count, st);
      return ;
    }

    // step 1. look up
    const int avg_token_per_expert = std::max(1, total_rows / num_experts);
    const int avg_token_per_expert_align = ((avg_token_per_expert + 15) / 16) * 16;

    std::pair<int, int> nk(gemm_n, gemm_k);
    std::pair<int, std::pair<int, int>> shape_mnk(avg_token_per_expert_align, nk);
    auto it = special_kernels_.find(shape_mnk);
    if(it != special_kernels_.end()) {
      // printf("find kernel for: %d,%d,%d.\n", avg_token_per_expert_align, gemm_n, gemm_k);
      it->second->Launch(A, B, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, ws, workspace_size, sm_count, st);
      return ;
    }
    /// step 2. look bs
    auto it_tile_n = tile_n_kernels_.find(avg_token_per_expert_align);
    if (it_tile_n != tile_n_kernels_.end()) {
      // printf("find tilen config. avg_token_per_expert_align\n");
      it_tile_n->second->Launch(A, B, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, ws, workspace_size, sm_count, st);
      return ;
    }

    /// step 3. use default config
    kernel_->Launch(A, B, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, ws, workspace_size, sm_count, st);
  }

  void RunV2(const InputType       *A,
          const InputType        *B,
          OutputType              *C,
          const GroupType  *total_rows_before_expert,
          int            total_rows,
          int64_t        gemm_m,
          int64_t        gemm_n,
          int64_t        gemm_k,
          int            num_experts,
          void           *ws,
          size_t         workspace_size,
          int            sm_count,
          cudaStream_t   st)
  {
    if (sm_count > 78) {
      h800_kernel_v2_->LaunchV2(A, B, C, total_rows_before_expert, total_rows, gemm_m, gemm_n, gemm_k, num_experts, ws, workspace_size, sm_count, st);
      return ;
    }

    /// step 1. look up
    const int avg_token_per_expert = std::max(1, total_rows / num_experts);
    const int avg_token_per_expert_align = ((avg_token_per_expert + 15) / 16) * 16;

    std::pair<int, int> nk(gemm_n, gemm_k);
    std::pair<int, std::pair<int, int>> shape_mnk(avg_token_per_expert_align, nk);
    auto it = special_kernels_v2_.find(shape_mnk);
    if(it != special_kernels_v2_.end()) {
      // printf("find kernel for: %d,%d,%d.\n", avg_token_per_expert_align, gemm_n, gemm_k);
      it->second->LaunchV2(A, B, C, total_rows_before_expert, total_rows, gemm_m, gemm_n, gemm_k, num_experts, ws, workspace_size, sm_count, st);
      return ;
    }
    /// step 2. look bs
    auto it_tile_n = tile_n_kernels_v2_.find(avg_token_per_expert_align);
    if (it_tile_n != tile_n_kernels_v2_.end()) {
      // printf("find tilen config. avg_token_per_expert_align\n");
      it_tile_n->second->LaunchV2(A, B, C, total_rows_before_expert, total_rows, gemm_m, gemm_n, gemm_k, num_experts, ws, workspace_size, sm_count, st);
      return ;
    }

    /// step 3. use default config
    kernel_v2_->LaunchV2(A, B, C, total_rows_before_expert, total_rows, gemm_m, gemm_n, gemm_k, num_experts, ws, workspace_size, sm_count, st);
  }

  size_t GetWorkSpace(int64_t gemm_n,
                      int64_t gemm_k,
                      int64_t total_rows,
                      int num_experts,
                      int sm_count)
  {
    return kernel_->GetWorkSpace(gemm_n, gemm_k, total_rows, num_experts, sm_count);
  }

  Impl() : kernel_(std::make_unique<DefaultKernel>()), kernel_v2_(std::make_unique<DefaultKernelV2>()),
          h800_kernel_(std::make_unique<H800DefaultKernel>()), h800_kernel_v2_(std::make_unique<H800DefaultKernelV2>())
  {
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_end_);

    Generate<true, true>(special_kernels_, tile_n_kernels_, kernel_); // V1，prefixsum & swap
    Generate<false, true>(special_kernels_v2_, tile_n_kernels_v2_, kernel_v2_); // V2, no prefixsum & swap
  }

  ~Impl()
  {
    cudaEventDestroy(ev_end_);
    cudaEventDestroy(ev_start_);
  }

  Kernel kernel_, kernel_v2_, h800_kernel_, h800_kernel_v2_;

  SpecialKernels special_kernels_, special_kernels_v2_;
  TileNKernels tile_n_kernels_, tile_n_kernels_v2_;

  cudaEvent_t ev_start_{};
  cudaEvent_t ev_end_{};
};


template <typename InputType, typename OutputType, typename GroupType>
HopperGmm<InputType, OutputType, GroupType>::HopperGmm() :
        impl_(std::make_unique<Impl>()){
}

template <typename InputType, typename OutputType, typename GroupType>
HopperGmm<InputType, OutputType, GroupType>::~HopperGmm() = default;

template <typename InputType, typename OutputType, typename GroupType>
void HopperGmm<InputType, OutputType, GroupType>::Run(const InputType         *A,
                        const InputType        *B,
                        OutputType              *C,
                        const GroupType  *total_rows_before_expert,
                        int            total_rows,
                        int64_t        gemm_n,
                        int64_t        gemm_k,
                        int            num_experts,
                        void           *ws,
                        size_t         workspace_size,
                        int            sm_count,
                        cudaStream_t   st)
{
  impl_->Run(A,
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
void HopperGmm<InputType, OutputType, GroupType>::RunV2(const InputType         *A,
                        const InputType        *B,
                        OutputType              *C,
                        const GroupType  *total_rows_before_expert,
                        int            total_rows,
                        int64_t        gemm_m,
                        int64_t        gemm_n,
                        int64_t        gemm_k,
                        int            num_experts,
                        void           *ws,
                        size_t         workspace_size,
                        int            sm_count,
                        cudaStream_t   st)
{
  impl_->RunV2(A,
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
size_t HopperGmm<InputType, OutputType, GroupType>::GetWorkSpace(int64_t gemm_n,
                                  int64_t gemm_k,
                                  int64_t total_rows,
                                  int num_experts,
                                  int sm_count)
{
  return impl_->GetWorkSpace(gemm_n,
                            gemm_k,
                            total_rows,
                            num_experts,
                            sm_count);
}

template class HopperGmm<cutlass::bfloat16_t, cutlass::bfloat16_t, int32_t>;

} // namespace aok