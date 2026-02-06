#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <mscclpp/concurrency_device.hpp>

#include "src/common/debug.cuh"
#include "src/communication/reducescatter_runner.cuh"
#include "src/common/cuda_utils.cuh"

namespace eps {

namespace {
using PackedFloat = union {
  int4 packed[2];
  float2 unpacked[4];
};

using PackedHalf = union {
  int4 packed;
  half2 unpacked[4];
};

using PackedBFloat16 = union {
  int4 packed;
  __nv_bfloat162 unpacked[4];
};

template <typename T>
struct PackedOn16Bytes;

template <>
struct PackedOn16Bytes<float> {
  using Type = PackedFloat;
  using UnpackedType = float2;
};

template <>
struct PackedOn16Bytes<half> {
  using Type = PackedHalf;
  using UnpackedType = half2;
};

template <>
struct PackedOn16Bytes<__nv_bfloat16> {
  using Type = PackedBFloat16;
  using UnpackedType = __nv_bfloat162;
};

template <typename Acc, typename T>
__forceinline__ __device__ void add128b_inplace(Acc &acc, T &b) {
  acc.unpacked[0] = acc.unpacked[0] + cuda_cast<float2>(b.unpacked[0]);
  acc.unpacked[1] = acc.unpacked[1] + cuda_cast<float2>(b.unpacked[1]);
  acc.unpacked[2] = acc.unpacked[2] + cuda_cast<float2>(b.unpacked[2]);
  acc.unpacked[3] = acc.unpacked[3] + cuda_cast<float2>(b.unpacked[3]);
}
}  // namespace

__device__ mscclpp::DeviceSyncer g_reducescatter_device_syncer;

template <typename T>
__global__ void ReduceScatterKernel(ReduceScatterKernelParams p) {
  static constexpr int ELTS_PER_ACCESS = 16 / sizeof(T);

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  const int tp_rank = p.global_rank % p.attn_tp_size;

  const size_t chan_offset = p.attn_tp_size * blockIdx.x;
  auto sm_chans = p.sm_channels + chan_offset;

  using PackedStruct = typename PackedOn16Bytes<T>::Type;
  T *buff = reinterpret_cast<T *>(sm_chans[tp_rank].src_);
  int4 *buff_packed = reinterpret_cast<int4 *>(buff);

  if (blockIdx.x == 0 && threadIdx.x < p.attn_tp_size && threadIdx.x != tp_rank) {
    sm_chans[threadIdx.x].signal();
    sm_chans[threadIdx.x].wait();
  }
  g_reducescatter_device_syncer.sync(gridDim.x);

  const size_t elems_to_recv_packed = p.elems_to_recv / ELTS_PER_ACCESS;
  const size_t index_offset_packed = p.local_elems_offset / ELTS_PER_ACCESS;
  for (size_t i = tid; i < elems_to_recv_packed; i += nthreads) {
    PackedFloat sum;
    sum.packed[0] = {0, 0, 0, 0};
    sum.packed[1] = {0, 0, 0, 0};

    for (int j = 0; j < p.attn_tp_size; j++) {
      int tp_peer_rank = j % p.attn_tp_size;

      PackedStruct val;
      val.packed = sm_chans[tp_peer_rank].read<int4>(index_offset_packed + i);
      add128b_inplace(sum, val);
    }
    PackedStruct res;
    res.unpacked[0] = cuda_cast<typename PackedOn16Bytes<T>::UnpackedType>(sum.unpacked[0]);
    res.unpacked[1] = cuda_cast<typename PackedOn16Bytes<T>::UnpackedType>(sum.unpacked[1]);
    res.unpacked[2] = cuda_cast<typename PackedOn16Bytes<T>::UnpackedType>(sum.unpacked[2]);
    res.unpacked[3] = cuda_cast<typename PackedOn16Bytes<T>::UnpackedType>(sum.unpacked[3]);
    buff_packed[index_offset_packed + i] = res.packed;
  }

  const size_t remain_elems = p.elems_to_recv % ELTS_PER_ACCESS;
  for (size_t i = tid; i < remain_elems; i += nthreads) {
    float sum = 0;
    for (int j = 0; j < p.attn_tp_size; j++) {
      int tp_peer_rank = j % p.attn_tp_size;

      T val = sm_chans[tp_peer_rank].read<T>(p.local_elems_offset + elems_to_recv_packed * ELTS_PER_ACCESS + i);
      sum += cuda_cast<float>(val);
    }
    reinterpret_cast<T *>(buff)[p.local_elems_offset + elems_to_recv_packed * ELTS_PER_ACCESS + i] = cuda_cast<T>(sum);
  }

  // buffer protection
  g_reducescatter_device_syncer.sync(gridDim.x);
  if (blockIdx.x == 0 && threadIdx.x < p.attn_tp_size && threadIdx.x != tp_rank) {
    sm_chans[threadIdx.x].signal();
    sm_chans[threadIdx.x].wait();
  }

}

template <typename T>
void ReduceScatter(ReduceScatterKernelParams p, cudaStream_t stream) {
  int nthreads = 512;
  ReduceScatterKernel<T><<<p.num_blocks, nthreads, 0, stream>>>(p);
}

template <>
void ReduceScatter<float>(ReduceScatterKernelParams p, cudaStream_t stream) {
  throw std::runtime_error("NOT IMPLEMENTED!");
}

template void ReduceScatter<half>(ReduceScatterKernelParams p, cudaStream_t stream);

#ifdef ENABLE_BF16
template void ReduceScatter<__nv_bfloat16>(ReduceScatterKernelParams p, cudaStream_t stream);
#endif

}  // namespace eps
