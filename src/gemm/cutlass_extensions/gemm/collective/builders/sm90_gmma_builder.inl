/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int capacity_bytes, class ElementA, class ElementB, class TileShapeMNK, int stages, int alignment = 128>
constexpr int
compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int capacity_bytes, class ElementA, class ElementB, class TileShapeMNK, int stages, int alignment = 128>
constexpr int
compute_stage_count_or_override(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int capacity_bytes_, class ElementA, class ElementB, class TileShapeMNK, int carveout_bytes_, int alignment = 128>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes_> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes_ =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
    static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity in gemm of blockwise/groupwise scale.
template<int capacity_bytes_, class ElementA, class ElementB, class ElementBlockScale, class TileShapeMNK, int ScaleMsPerTile, int ScaleNsPerTile, int carveout_bytes_, int alignment = 128>
constexpr int
compute_stage_count_with_blockwise_scale(StageCountAutoCarveout<carveout_bytes_> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto scale_bits = cute::sizeof_bits_v<ElementBlockScale>;
  constexpr int stage_bytes_ =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(scale_bits * ScaleMsPerTile) + // scale of tensor A
    cutlass::bits_to_bytes(scale_bits * ScaleNsPerTile);  // scale of tensor B

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
    static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an optional scale matrix), or overrides with manual count.
template<int capacity_bytes, class ElementA, class ElementB, class ElementScale, class ElementZero, class TileShapeMNK, int stages, int alignment = 128>
constexpr int
compute_stage_count_or_override_single_affine_transformed_input(StageCount<stages> stage_count) {
  return stages;
}

template <class Element>
constexpr int get_bits_for_possibly_void_element() {
  if constexpr (cute::is_same_v<Element, void>) {
    return 0;
  }
  else {
    return sizeof_bits<Element>::value;
  }
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an optional scale matrix), or overrides with manual count.
template<int capacity_bytes_, class ElementA, class ElementB, class ElementScale, class ElementZero, class TileShapeMNK, int carveout_bytes_, int alignment = 128>
constexpr int
compute_stage_count_or_override_single_affine_transformed_input(StageCountAutoCarveout<carveout_bytes_> stage_count) {

  // 32 bytes to account for barriers etc.
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr int scale_zero_k_tile = 1;
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto s_bits = get_bits_for_possibly_void_element<ElementScale>();
  constexpr auto z_bits = get_bits_for_possibly_void_element<ElementZero>();

  constexpr auto scale_bytes = cutlass::bits_to_bytes(s_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);
  constexpr auto zero_bytes  = cutlass::bits_to_bytes(z_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);
  static_assert(scale_bytes % 128 == 0, "Scale bytes must be a multiple of 128");
  static_assert(zero_bytes  % 128 == 0, "Zero bytes must be a multiple of 128");

  // When scales are void, s_bits will be 0 so no smem will be allocated for scales.
  constexpr int stage_bytes_ =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    scale_bytes + zero_bytes;

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
    static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB>
constexpr bool
is_swapAB(){
  constexpr bool IsInputSizeTwoBytes = is_input_size_two_bytes<ElementA, ElementB>();
  constexpr bool IsLayoutAkBmn = cutlass::gemm::detail::is_k_major_A<LayoutA>() &&
                                 cutlass::gemm::detail::is_mn_major_B<LayoutB>();
  constexpr bool SwapAB = !IsInputSizeTwoBytes && IsLayoutAkBmn;
  return SwapAB;
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class KernelScheduleType>
constexpr bool
is_warpspecialized_transpose_B(){
  constexpr bool IsInputSizeTwoBytes = is_input_size_two_bytes<ElementA, ElementB>();
  constexpr bool IsLayoutAmnBmn = cutlass::gemm::detail::is_mn_major_A<LayoutA>() &&
                                  cutlass::gemm::detail::is_mn_major_B<LayoutB>();
  constexpr bool IsWarpSpecialized = cute::is_base_of_v<KernelTmaWarpSpecialized, KernelScheduleType>                ||
                                     cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, KernelScheduleType>        ||
                                     cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, KernelScheduleType>     ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecialized, KernelScheduleType>            ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecializedPingpong, KernelScheduleType>    ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecializedCooperative, KernelScheduleType>;
  constexpr bool IsWarpSpecializedTransposeB = !IsInputSizeTwoBytes && IsLayoutAmnBmn && IsWarpSpecialized;
  return IsWarpSpecializedTransposeB;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_any_of_v<KernelScheduleType,
                         KernelTmaWarpSpecialized,
                         KernelTmaWarpSpecializedCooperative,
                         KernelTmaWarpSpecializedPingpong,
                         KernelPtrArrayTmaWarpSpecializedCooperative,
                         KernelPtrArrayTmaWarpSpecializedPingpong>) &&
       not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");

  static constexpr bool IsArrayOfPointersGemm = (cute::is_any_of_v<KernelScheduleType,
                                                                   KernelPtrArrayTmaWarpSpecializedCooperative,
                                                                   KernelPtrArrayTmaWarpSpecializedPingpong>);
  static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();
  static_assert(!IsFP8Input || (IsFP8Input && !IsArrayOfPointersGemm),
                "KernelPtrArrayTmaWarpSpecialized[Cooperative|Pingpong] is only compatible with FP8 FastAccum version right now.");

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  static constexpr bool IsCooperative = cute::is_any_of_v<KernelScheduleType,
                                                          KernelTmaWarpSpecializedCooperative,
                                                          KernelPtrArrayTmaWarpSpecializedCooperative>;
  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 2 /* for A and B */ : 0;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage);

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes - KernelSmemCarveout,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
      MainloopSm90ArrayTmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
      /* For FP8 use a separate mainloop compared to other datatypes */
      cute::conditional_t<IsFP8Input,
          MainloopSm90TmaGmmaWarpSpecializedFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
          MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>>;

  using SmemCopyAtomA = void; 
  using SmemCopyAtomB = void; 

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

}