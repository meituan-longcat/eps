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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIaok_group_array_problem_shape.hppBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief This file contains definitions and utility functions for describing problem shapes 
           for 3.x Ptr-Array GEMMs and Grouped GEMMs.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_coord.h"

#include "cute/container/array.hpp"

#if ! defined(__CUDACC_RTC__)
#include <initializer_list>
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////
/* 
  ProblemShape: Shape<int,int,int> or Shape<int,int,int,int>
  GroupType: 表明当前的batch_size or batch_size_prefixsum是int32_t还是int64_t
  IsPrefixSum: 表明当前group_list信息
  IsSwapAB: 表明当前是否交换AB； TODO: 当前优先级不高
 */
template <typename ProblemShape_, typename GroupType_, bool IsPrefixSum_=true, bool IsSwapAB_=true>
struct AokGroupProblemShape {
  using UnderlyingProblemShape = ProblemShape_; // shape<int,int,int,int>
  using GroupType = GroupType_;

  static const bool IsPrefixSum = IsPrefixSum_;
  static const bool IsSwapAB = IsSwapAB_;

  // using UnderlyingProblemShape = int32_t; // 这里只提供一个tokens_per_expert
  UnderlyingProblemShape problem_sizes{-1,-1,-1,-1};
  const GroupType* tokens_per_expert = nullptr;
  int32_t num_groups = 1;

  CUTLASS_HOST_DEVICE
  AokGroupProblemShape() {}

  CUTLASS_HOST_DEVICE
  AokGroupProblemShape(const GroupType* tokens_per_expert, UnderlyingProblemShape problem_sizes, int32_t num_groups): 
            tokens_per_expert(tokens_per_expert), problem_sizes(problem_sizes), num_groups(num_groups) {}

  CUTLASS_HOST_DEVICE
  int32_t groups() const { return num_groups; }

  // only work on device,
  CUTLASS_HOST_DEVICE
  UnderlyingProblemShape
  get_problem_shape(int32_t group_idx) const {
    // return problem_shapes[group_idx];
    // swap AB
    /* 
        problem_sizes: gemm_m, gemm_n, gemm_k, groups，为未swapab的值
     */
    if constexpr (IsPrefixSum) {
      // old version
      if constexpr (IsSwapAB) {
        return {get<1>(problem_sizes), tokens_per_expert[group_idx+1] - tokens_per_expert[group_idx], get<2>(problem_sizes), get<3>(problem_sizes)};
      }
      else {
        return {tokens_per_expert[group_idx+1] - tokens_per_expert[group_idx], get<1>(problem_sizes), get<2>(problem_sizes), get<3>(problem_sizes)};
      }
    }
    else {
      // new version deepep
      if constexpr (IsSwapAB) {
        return {get<1>(problem_sizes), tokens_per_expert[group_idx], get<2>(problem_sizes), get<3>(problem_sizes)};
      }
      else {
        return {tokens_per_expert[group_idx], get<1>(problem_sizes), get<2>(problem_sizes), get<3>(problem_sizes)};
      }
    }
    
  }

  // no host info
  // CUTLASS_HOST_DEVICE
  // UnderlyingProblemShape
  // get_host_problem_shape(int32_t group_idx) const {
  //   return false;
  // }

  CUTLASS_HOST_DEVICE
  bool
  is_host_problem_shape_available() {
    return false;
  }
};

} // namespace cutlass::gemm 
