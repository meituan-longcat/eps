#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace eps {

template <typename T>
inline T *get_ptr(at::Tensor &t) {
  return reinterpret_cast<T *>(t.data_ptr());
}

template <typename T>
inline const T *get_ptr(const at::Tensor &t) {
  return reinterpret_cast<const T *>(t.data_ptr());
}

} // namespace eps

#define EPS_CHECK_TYPE(x, st) \
  TORCH_CHECK((x).scalar_type() == (st), "Inconsistency of Tensor type: " #x)

#define EPS_CHECK_TH_CUDA(x) \
  TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

#define EPS_CHECK_CPU(x) \
  TORCH_CHECK(!(x).is_cuda(), #x " must be a CPU tensor")

#define EPS_CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define EPS_CHECK_INPUT(x, st) \
  do {                         \
    EPS_CHECK_TH_CUDA(x);      \
    EPS_CHECK_CONTIGUOUS(x);   \
    EPS_CHECK_TYPE(x, st);     \
  } while (0)

#define EPS_CHECK_TENSOR(ndim, x)                                                                 \
  do {                                                                                             \
    TORCH_CHECK((x).is_cuda(), "Tensor " #x " must be on GPU");                                    \
    TORCH_CHECK((x).is_contiguous(), "Tensor " #x " must be contiguous");                          \
    TORCH_CHECK((x).dim() == (ndim), "Tensor " #x " must be ", (ndim), " dimensional");            \
  } while (0)
