#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <mscclpp/concurrency_device.hpp>

#include "src/communication/allgather_runner.cuh"

namespace eps {

__device__ mscclpp::DeviceSyncer g_allgather_device_syncer;

template <typename T>
__global__ void __launch_bounds__(1024, 1) AllGatherKernel(AllGatherKernelParams p) {
  static constexpr int WARP_SIZE = 32;

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nthreads = blockDim.x * gridDim.x;
  const size_t nwarp = nthreads / WARP_SIZE;
  const size_t npeer = p.attn_tp_size - 1;
  const size_t chan_offset = p.attn_tp_size * blockIdx.x;
  const int tp_rank = p.global_rank % p.attn_tp_size;

  auto sm_chans = p.sm_channels + chan_offset;

  if (threadIdx.x < p.attn_tp_size && threadIdx.x != tp_rank) {
    sm_chans[threadIdx.x].signal();
    sm_chans[threadIdx.x].wait();
  }
  __syncthreads();

  const size_t bytes_to_send = p.elems_to_send * sizeof(T);
  const size_t local_data_offset = p.local_elems_offset * sizeof(T);
  const size_t bytes = bytes_to_send * npeer;
  size_t unit_bytes_per_thread;
  if (bytes >= nthreads * 64) {
    unit_bytes_per_thread = 64;
  } else {
    unit_bytes_per_thread = 16;
  }
  const size_t unit_bytes_per_warp = unit_bytes_per_thread * WARP_SIZE;
  const size_t unit_bytes = unit_bytes_per_warp * nwarp;
  const size_t nloop = bytes / unit_bytes;

  auto get_peer_idx = [=](size_t id) { 
    return id >= tp_rank ? id + 1 : id; 
  };

  if (nloop > 0) {
    const size_t peer_idx = get_peer_idx(wid % npeer);
    const size_t offset = local_data_offset + (wid / npeer) * unit_bytes_per_warp;
    sm_chans[peer_idx].put<16, false>(offset, unit_bytes_per_warp, lid, WARP_SIZE);
  }

  for (size_t i = 1; i < nloop; ++i) {
    const size_t gwid = wid + i * nwarp;
    const size_t peer_idx = get_peer_idx(gwid % npeer);
    const size_t offset = local_data_offset + (gwid / npeer) * unit_bytes_per_warp;
    sm_chans[peer_idx].put<16, false>(offset, unit_bytes_per_warp, lid, WARP_SIZE);
  }

  if (bytes % unit_bytes > 0) {
    const size_t gwid = wid + nloop * nwarp;
    const size_t peer_idx = get_peer_idx(gwid % npeer);
    const size_t offset_within_rank = (gwid / npeer) * unit_bytes_per_warp;
    const size_t offset = local_data_offset + offset_within_rank;
    const size_t remain_bytes = (offset_within_rank + unit_bytes_per_warp > bytes_to_send)
                                    ? ((bytes_to_send > offset_within_rank) ? (bytes_to_send - offset_within_rank) : 0)
                                    : unit_bytes_per_warp;
    if (remain_bytes > 0) {
      sm_chans[peer_idx].put<16, true>(offset, remain_bytes, lid, WARP_SIZE);
    }
  }

  g_allgather_device_syncer.sync(gridDim.x);

  if (threadIdx.x < p.attn_tp_size && threadIdx.x != tp_rank) {
    sm_chans[threadIdx.x].signal();
    sm_chans[threadIdx.x].wait();
  }
}

template <typename T>
void AllGather(AllGatherKernelParams p, cudaStream_t stream) {
  if (p.elems_to_send * sizeof(T) % 16 != 0) {
    throw std::runtime_error(
        "Not supported alignment: " + std::to_string(p.elems_to_send) + " * " + std::to_string(sizeof(T)));
  }

  AllGatherKernel<T><<<p.num_blocks, 1024, 0, stream>>>(p);
}

template void AllGather<float>(AllGatherKernelParams p, cudaStream_t stream);

template void AllGather<half>(AllGatherKernelParams p, cudaStream_t stream);

#ifdef ENABLE_BF16
template void AllGather<__nv_bfloat16>(AllGatherKernelParams p, cudaStream_t stream);
#endif
}  // namespace eps
