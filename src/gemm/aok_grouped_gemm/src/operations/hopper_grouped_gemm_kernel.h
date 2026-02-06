#pragma once

#include <algorithm>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "src/gemm/cutlass_extensions/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "src/gemm/cutlass_extensions/gemm/collective/collective_builder.hpp"
#include "src/gemm/cutlass_extensions/epilogue/collective/collective_builder.hpp"
#include "src/gemm/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp"
#include "src/gemm/cutlass_extensions/gemm/device/gemm_universal_adapter.h"
#include "src/gemm/cutlass_extensions/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "gemm/aok_group_array_problem_shape.hpp"

#include "common/helper.h"

namespace aok {

using namespace cute;
using cutlass::layout::RowMajor;
using cutlass::layout::ColumnMajor;

template<typename InputType,
        typename OutputType,
        typename GroupType>
struct Base_SM90_GROUPED_GEMM {
  virtual ~Base_SM90_GROUPED_GEMM() = default;

  virtual void Launch(const InputType *A,
                    const InputType   *B,
                    OutputType        *C,
                    const GroupType     *total_rows_before_expert,
                    int               total_rows,
                    int64_t           gemm_n,
                    int64_t           gemm_k,
                    int               num_experts,
                    void              *ws,
                    size_t            workspace_size,
                    int               sm_count,
                    cudaStream_t      st) = 0;
  
  virtual void LaunchV2(const InputType *A,
                    const InputType   *B,
                    OutputType        *C,
                    const GroupType     *total_rows_before_expert,
                    int               total_rows,
                    int64_t           gemm_m,
                    int64_t           gemm_n,
                    int64_t           gemm_k,
                    int               num_experts,
                    void              *ws,
                    size_t            workspace_size,
                    int               sm_count,
                    cudaStream_t      st) = 0;

  virtual size_t GetWorkSpace(int64_t gemm_n, int64_t gemm_k, int64_t total_rows, int num_experts, int sm_count) = 0;
};

namespace detail {
template <typename InputType, typename OutputType, typename Offset, typename StrideA, typename StrideB, typename StrideC, typename GroupType>
__global__ void DoOffsetComputeKernel(const InputType *A, const InputType *B, OutputType *C, const GroupType *total_rows_before_expert, void *ws, int gemm_m, int gemm_k, int num_experts)
{
  if(threadIdx.x == 0) {
    /// step 1. make problem_sizes
    Offset offset(ws, num_experts);
    auto *problem_sizes = offset.problem_sizes;

    for(int i = 0; i < num_experts; ++i)
    {
      problem_sizes[i] = {gemm_m, total_rows_before_expert[i+1] - total_rows_before_expert[i], gemm_k};
      // test no exchange
      // problem_sizes[i] = {total_rows_before_expert[i+1] - total_rows_before_expert[i], gemm_m, gemm_k};

      // printf("group[%d], mnk: %d,%d,%d.\n",
      //     i, static_cast<int>(total_rows_before_expert[i+1] - total_rows_before_expert[i]),
      //     gemm_m, gemm_k);
    }
    
    // offset.print();

    int64_t offset_A, offset_B, offset_C;

    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;

    auto *stride_A = offset.stride_A;
    auto *stride_B = offset.stride_B;
    auto *stride_C = offset.stride_C;
    auto **list_A = offset.list_A;
    auto **list_B = offset.list_B;
    auto **list_C = offset.list_C;

    for(int i = 0; i < num_experts; ++i)
    {
      auto problem = problem_sizes[i];
      auto M = static_cast<int64_t>(get<0>(problem));
      auto N = static_cast<int64_t>(get<1>(problem));
      auto K = static_cast<int64_t>(get<2>(problem));

      offset_A = total_elements_A;
      offset_B = total_elements_B;
      offset_C = total_elements_C;

      int64_t elements_A = M * K;
      int64_t elements_B = K * N;
      int64_t elements_C = M * N;

      total_elements_A += elements_A;
      total_elements_B += elements_B;
      total_elements_C += elements_C;

      stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, Int<1>{}});
      stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, Int<1>{}});
      stride_C[i] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, Int<1>{}});

      // printf("[%d,%d,%d][%d,%d,%d], offset_A/B/C: %ld,%ld,%ld, elements_C: %ld, M,N,K: %ld,%ld,%ld\n",
      //     blockIdx.x, blockIdx.y, blockIdx.z,
      //     threadIdx.x, threadIdx.y, threadIdx.z,
      //     offset_A, offset_B, offset_C, elements_C, M, N,K);

      list_A[i] = (InputType *)(A + offset_A);
      list_B[i] = (InputType *)(B + offset_B);
      list_C[i] = (OutputType *)(C + offset_C);
    }
  }
}

template <typename InputType, typename OutputType, typename Offset, typename StrideA, typename StrideB, typename StrideC, typename GroupType>
__global__ void DoOffsetComputeKernelV2(const InputType *A, const InputType *B, OutputType *C, const GroupType *total_rows_before_expert, void *ws, int gemm_m, int gemm_n, int gemm_k, int num_experts)
{
  if(threadIdx.x == 0) {
    /// step 1. make problem_sizes
    Offset offset(ws, num_experts);
    auto *problem_sizes = offset.problem_sizes;

    printf("value is: %d.\n", (0+ 0 * (16-1))/ 16 * 16);

    for(int i = 0; i < num_experts; ++i)
    {
      problem_sizes[i] = {gemm_m, total_rows_before_expert[i+1] - total_rows_before_expert[i], gemm_k};
      // test no exchange
      // problem_sizes[i] = {total_rows_before_expert[i+1] - total_rows_before_expert[i], gemm_m, gemm_k};

      // printf("group[%d], mnk: %d,%d,%d.\n",
      //     i, static_cast<int>(total_rows_before_expert[i+1] - total_rows_before_expert[i]),
      //     gemm_m, gemm_k);
    }
    
    // offset.print();

    int64_t offset_A, offset_B, offset_C;

    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;

    auto *stride_A = offset.stride_A;
    auto *stride_B = offset.stride_B;
    auto *stride_C = offset.stride_C;
    auto **list_A = offset.list_A;
    auto **list_B = offset.list_B;
    auto **list_C = offset.list_C;

    for(int i = 0; i < num_experts; ++i)
    {
      auto problem = problem_sizes[i];
      auto M = static_cast<int64_t>(get<0>(problem));
      // auto N = static_cast<int64_t>(get<1>(problem));
      auto N = static_cast<int64_t>(gemm_n);
      auto K = static_cast<int64_t>(get<2>(problem));

      offset_A = total_elements_A;
      offset_B = total_elements_B;
      offset_C = total_elements_C;

      int64_t elements_A = M * K;
      int64_t elements_B = K * N;
      int64_t elements_C = M * N;

      printf("expert: %d, M,N,K: %d,%d,%d.\n",
          i, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
      printf("expert: %d, elements_A,elements_B,elements_C: %d,%d,%d.\n",
          i, static_cast<int>(elements_A), static_cast<int>(elements_B), static_cast<int>(elements_C));

      total_elements_A += elements_A;
      total_elements_B += elements_B;
      total_elements_C += elements_C;

      stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, Int<1>{}});
      stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, Int<1>{}});
      stride_C[i] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, Int<1>{}});

      // printf("[%d,%d,%d][%d,%d,%d], offset_A/B/C: %ld,%ld,%ld, elements_C: %ld, M,N,K: %ld,%ld,%ld\n",
      //     blockIdx.x, blockIdx.y, blockIdx.z,
      //     threadIdx.x, threadIdx.y, threadIdx.z,
      //     offset_A, offset_B, offset_C, elements_C, M, N,K);

      list_A[i] = (InputType *)(A + offset_A);
      list_B[i] = (InputType *)(B + offset_B);
      list_C[i] = (OutputType *)(C + offset_C);
    }
  }
}


} // namespace detail
// sm90 impl
template<typename LayoutA_,
        typename LayoutB_,
        typename LayoutC_,
        typename InputType_,
        typename OutputType_,
        typename Accumulator_,
        typename TileShape_,
        typename ClusterShape_,
        typename Stages_,
        typename GroupType_,
        bool IsPrefixSum_,
        bool IsSwapAB_>
struct SM90_GROUPED_GEMM
    : public Base_SM90_GROUPED_GEMM<InputType_, OutputType_, GroupType_>{
  
  // static_assert(cute::is_same<LayoutA_, RowMajor>::value
  //                 & cute::is_same<LayoutB_, ColumnMajor>::value
  //                 & cute::is_same<LayoutC_, ColumnMajor>::value,
  //                 "Currently, A Matrix must be RowMajor, B/C Matrix must be all ColumnMajor");
  static_assert(cute::is_same<InputType_, OutputType_>::value, "InputType_ and OutputType_ must be same type.");
  static_assert(cute::is_same<InputType_, cutlass::half_t>::value
                    | cute::is_same<InputType_, cutlass::bfloat16_t>::value,
                    "InputType must be cutlass::half_t or cutlass::bfloat16_t");
 
  using ElementA = InputType_;
  using ElementB = InputType_;
  using ElementC = OutputType_;
  using InputType = InputType_;
  using OutputType = OutputType_;
  using GroupType = GroupType_;

  static const bool IsPrefixSum = IsPrefixSum_;
  static const bool IsSwapAB = IsSwapAB_;
  using ProblemShape = cutlass::gemm::AokGroupProblemShape<Shape<int, int, int, int>, GroupType, IsPrefixSum, IsSwapAB>;

  // A matrix configuration
  using         LayoutA     = cute::conditional_t<IsSwapAB, typename cutlass::layout::LayoutTranspose<LayoutB_>::type, LayoutA_>;                      // Layout type for A matrix operand
  static constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         LayoutB     = cute::conditional_t<IsSwapAB, typename cutlass::layout::LayoutTranspose<LayoutA_>::type, LayoutB_>;                  // Layout type for B matrix operand
  static constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         LayoutC     = cute::conditional_t<IsSwapAB, typename cutlass::layout::LayoutTranspose<LayoutC_>::type, LayoutC_>;                   // Layout type for C and D matrix operands
  static constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Alignment of C matrix in units of elements (up to 16 bytes)

  // Core kernel configurations
  using ElementAccumulator  = Accumulator_;                                          // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  // using TileShape           = TileShape_;                           // Threadblock-level tile size
  // noswapAB场景,切分先fix 128x128
  using TileShape = cute::conditional_t<IsSwapAB, TileShape_, Shape<_128, _128, _64>>;
  using ClusterShape        = ClusterShape_;                                // Shape of the threadblocks in a cluster
  using Stages = Stages_;

  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  // // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;                     // Epilogue to launch
  // // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;

  // using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  // // using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  // using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;


  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      // Shape<int,int,int,int>,
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  SM90_GROUPED_GEMM() {}

  void Launch(const InputType       *A,
              const InputType       *B,
              OutputType            *C,
              const GroupType       *total_rows_before_expert,
              int                   total_rows,
              int64_t               gemm_n,
              int64_t               gemm_k,
              int                   num_experts,
              void                  *ws,
              size_t                workspace_size,
              int                   sm_count,
              cudaStream_t          st) override
  {
    Gemm gemm;
    cutlass::KernelHardwareInfo hw_info;
    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    hw_info.device_id = 0;
    hw_info.sm_count = sm_count;


    // ProblemShapeType problem_size{};
    using UnderlyingProblemShape = typename ProblemShapeType::UnderlyingProblemShape;
    // v1(old version) launch, no gemm_m
    int64_t gemm_m = static_cast<int>(total_rows);
    auto probelm_mnkl = UnderlyingProblemShape{
        static_cast<int>(gemm_m),
        static_cast<int>(gemm_n),
        static_cast<int>(gemm_k),
        num_experts};
    ProblemShapeType problem_size = {
      total_rows_before_expert,
      probelm_mnkl, // swap在kernel内去做
      num_experts
    };
 
    if constexpr (IsSwapAB) {
      std::swap(gemm_m, gemm_n);
      std::swap(A, B);
      // TODO: swap ptr_A and ptr_B
    }
    
    // swap AB
    // gemm_m实际并不会导致stride错误，（layout决定）
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, {static_cast<int>(gemm_m), static_cast<int>(gemm_k), num_experts});
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, {static_cast<int>(gemm_n), static_cast<int>(gemm_k), num_experts});
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, {static_cast<int>(gemm_m), static_cast<int>(gemm_n), num_experts});

    typename Gemm::Arguments arguments{
      // cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmUniversalMode::kGrouped,
      problem_size,
      {(const ElementA*)A, stride_A, (const ElementB*)B, stride_B},
      {{1.0, 0.0}, // epilogue.thread
       (const ElementC*)C, stride_C, (ElementC*)C, stride_C},
      hw_info,
      {}, // scheduler
      // total_rows_before_expert
    };

    // workspace need
    size_t ws_size = Gemm::get_workspace_size(arguments);

    assert(ws_size <= workspace_size);

    // // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));
    // Initialize CUTLASS kernel with arguments and workspace pointer
    // 实际不走ptr_array,是不需要workspace,因此init workspace 直接pass // TODO: check
    // 这里实际就GemmKernel to_underlying_arguments
    CUTLASS_CHECK(gemm.initialize(arguments, ws));
    
    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run(st));

  }

  void LaunchV2(const InputType       *A,
              const InputType       *B,
              OutputType            *C,
              const GroupType         *total_rows_before_expert,
              int                   total_rows,
              int64_t               gemm_m,
              int64_t               gemm_n,
              int64_t               gemm_k,
              int                   num_experts,
              void                  *ws,
              size_t                workspace_size,
              int                   sm_count,
              cudaStream_t          st) override
  {
    Gemm gemm;
    cutlass::KernelHardwareInfo hw_info;
    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    hw_info.device_id = 0;
    hw_info.sm_count = sm_count;


    // ProblemShapeType problem_size{};
    using UnderlyingProblemShape = typename ProblemShapeType::UnderlyingProblemShape;
    auto probelm_mnkl = UnderlyingProblemShape{
        static_cast<int>(gemm_m),
        static_cast<int>(gemm_n),
        static_cast<int>(gemm_k),
        num_experts};
    ProblemShapeType problem_size = {
      total_rows_before_expert,
      probelm_mnkl, // swap在kernel内去做
      num_experts
    };
 
    if constexpr (IsSwapAB) {
      std::swap(gemm_m, gemm_n);
      std::swap(A, B);
    }
    
    // swap AB
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, {static_cast<int>(gemm_m), static_cast<int>(gemm_k), num_experts});
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, {static_cast<int>(gemm_n), static_cast<int>(gemm_k), num_experts});
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, {static_cast<int>(gemm_m), static_cast<int>(gemm_n), num_experts});

    typename Gemm::Arguments arguments{
      // cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmUniversalMode::kGrouped,
      problem_size,
      {(const ElementA*)A, stride_A, (const ElementB*)B, stride_B},
      {{1.0, 0.0}, // epilogue.thread
       (const ElementC*)C, stride_C, (ElementC*)C, stride_C},
      hw_info,
      {}, // scheduler
      // total_rows_before_expert
    };

    // workspace need
    size_t ws_size = Gemm::get_workspace_size(arguments);

    assert(ws_size <= workspace_size);

    // // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));
    // Initialize CUTLASS kernel with arguments and workspace pointer
    // 实际不走ptr_array,是不需要workspace,因此init workspace 直接pass // TODO: check
    // 这里实际就GemmKernel to_underlying_arguments
    CUTLASS_CHECK(gemm.initialize(arguments, ws));
    
    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run(st));
  }


  size_t GetWorkSpace(int64_t gemm_n, int64_t gemm_k, int64_t total_rows, int num_experts, int sm_count) override
  {

    cutlass::KernelHardwareInfo hw_info;
    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    hw_info.device_id = 0;
    hw_info.sm_count = sm_count;
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, {static_cast<int>(gemm_n), static_cast<int>(gemm_k), num_experts});
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, {static_cast<int>(total_rows), static_cast<int>(gemm_k), num_experts});
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, {static_cast<int>(gemm_n), static_cast<int>(total_rows), num_experts});
    
    typename Gemm::Arguments arguments{
      // cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {},
      {(const ElementA*)nullptr, stride_A, (const ElementB*)nullptr, stride_B},
      {{1.0, 0.0}, // epilogue.thread
       (const ElementC*)nullptr, stride_C, (ElementC*)nullptr, stride_C},
      hw_info,
      {}, // scheduler
      // total_rows_before_expert
    };
    return Gemm::get_workspace_size(arguments);;
    
  }

};

} // namespace aok
