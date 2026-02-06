#include "src/communication/alltoall_runner.cuh"
#include "src/common/debug.cuh"

#include <cub/cub.cuh>

namespace eps {

__device__ __inline__ bool is_same_node(int my_rank, int remote_rank, int num_ranks_per_node) {
  return (my_rank / num_ranks_per_node) == (remote_rank / num_ranks_per_node);
}

__device__ void allGatherSendPlanSameNode(CommPlanKernelParams p, int remoteRank) {
  int warpSize = 32;
  int laneIdx = threadIdx.x % warpSize;

  p.comm_buff.smChan_put(remoteRank,
      p.my_rank * p.ep_world_size * p.num_stages * sizeof(int),
      0,
      p.ep_world_size * p.num_stages * sizeof(int),
      laneIdx,
      warpSize,
      4);
  __syncwarp();

  if (laneIdx == 0) {
    p.comm_buff.smChan_signal(remoteRank);
    p.comm_buff.smChan_wait(remoteRank);
  }
}

__device__ void allGatherSendPlanCrossNode(CommPlanKernelParams p, int remoteRank) {
  p.comm_buff.proxyChan_putWithSignal(remoteRank,
      p.my_rank * p.ep_world_size * p.num_stages * sizeof(int),
      0,
      p.ep_world_size * p.num_stages * sizeof(int));
  p.comm_buff.proxyChan_wait(remoteRank);
}

__global__ void allGatherSendPlan(CommPlanKernelParams p) {
  int my_node = p.my_rank / p.num_ranks_per_node;
  int my_node_rank_start = my_node * p.num_ranks_per_node;
  int my_node_rank_end = (my_node + 1) * p.num_ranks_per_node;

  int warpSize = 32;
  int warpIdx = threadIdx.x / warpSize;

  if (warpIdx < p.num_ranks_per_node) {
      int remoteRank = my_node_rank_start + (p.my_rank - my_node_rank_start + warpIdx) % p.num_ranks_per_node;
      allGatherSendPlanSameNode(p, remoteRank);
  } else {
    int r = threadIdx.x - warpSize * p.num_ranks_per_node;
    if (r < (p.ep_world_size - p.num_ranks_per_node)) {
      int remoteRank = (my_node_rank_end + r) % p.ep_world_size;
      allGatherSendPlanCrossNode(p, remoteRank);
    }
  }
}

template <int BLOCK_SIZE>
__global__ void getRecvPlan(CommPlanKernelParams p) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(BLOCK_SIZE > p.ep_world_size * p.num_stages);

  extern __shared__ int num_tokens[]; /*ep_world_size, num_stages*/
  using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storge;

  for (int rank = 0; rank < p.ep_world_size; rank++) {
    for (int i = threadIdx.x; i < p.ep_world_size * p.num_stages; i += blockDim.x) {
      int stage_idx = i % p.num_stages;
      int device_idx = i / p.num_stages;
      int begin = (rank == 0 && stage_idx == 0)
          ? 0
          : p.gathered_send_plan[device_idx * p.ep_world_size * p.num_stages + rank * p.num_stages + stage_idx - 1];
      int end = p.gathered_send_plan[device_idx * p.ep_world_size * p.num_stages + rank * p.num_stages + stage_idx];
      num_tokens[i] = end - begin;
    }
    __syncthreads();

    int thread_data = (threadIdx.x < p.ep_world_size * p.num_stages) ? num_tokens[threadIdx.x] : 0;

    int thread_inclusive;
    BlockScan(temp_storge).InclusiveSum(thread_data, thread_inclusive);
    __syncthreads(); // Barrier for smem reuse
    if (rank == p.my_rank && threadIdx.x == p.ep_world_size * p.num_stages - 1 && p.plan.num_will_recv_unique_tokens != nullptr) {
      *p.plan.num_will_recv_unique_tokens = thread_inclusive;
    }

    int thread_exclusive;
    BlockScan(temp_storge).ExclusiveSum(thread_data, thread_exclusive);
    __syncthreads(); // Barrier for smem reuse

    if (p.my_rank * p.num_stages <= threadIdx.x && threadIdx.x < (p.my_rank + 1) * p.num_stages) {
      p.plan.remote_recv_plan[rank * p.num_stages + (threadIdx.x - p.my_rank * p.num_stages)] = thread_exclusive;
    }

    if (rank == p.my_rank && threadIdx.x < p.ep_world_size * p.num_stages) {
      p.plan.recv_plan[threadIdx.x] = thread_inclusive;
    }
  }
}

template<typename T>
__global__ void flushProxyChansKernel(FlushProxyChansParams<T> p) {
  int remoteRank = threadIdx.x;
  if (remoteRank < p.world_size && !is_same_node(p.my_rank, remoteRank, p.num_ranks_per_node)) {
    p.comm_buff.proxyChan_flush(remoteRank);
  }
  __syncthreads();
}

template<typename T>
void flushProxyChans(FlushProxyChansParams<T> p, cudaStream_t stream) {
  flushProxyChansKernel<<<1, p.world_size, 0, stream>>>(p);
}

void generateCommPlan(CommPlanKernelParams p, cudaStream_t stream) {
  allGatherSendPlan<<<1, p.num_ranks_per_node * 32 + (p.ep_world_size - p.num_ranks_per_node), 0, stream>>>(p);
  sync_check_cuda_error_eps();

  static constexpr int BLOCK = 512;
  if (BLOCK < p.ep_world_size * p.num_stages) {
    throw std::runtime_error("ep_world_size * num_stages is too big.");
  }

  getRecvPlan<BLOCK><<<1, BLOCK, sizeof(int) * p.ep_world_size * p.num_stages, stream>>>(p);
  sync_check_cuda_error_eps();
}

template <typename T, int Alignment>
__global__ void All2AllOneShotKernel(All2AllKernelParams<T> p) {
  int remoteRank = blockIdx.x;

  int recv_start = p.plan.remote_recv_plan[remoteRank * p.num_stages];
  int send_start = remoteRank == 0 ? 0 : p.plan.send_plan[remoteRank * p.num_stages - 1];
  int send_end = p.plan.send_plan[remoteRank * p.num_stages + p.num_stages - 1];

  if (is_same_node(p.my_rank, remoteRank, p.num_ranks_per_node)) {
    p.comm_buff.smChan_put(remoteRank,
        recv_start * p.cols * sizeof(T),
        send_start * p.cols * sizeof(T),
        (send_end - send_start) * p.cols * sizeof(T),
        threadIdx.x,
        blockDim.x,
        Alignment);
    __syncthreads();

    if (threadIdx.x == 0) {
      p.comm_buff.smChan_signal(remoteRank);
      p.comm_buff.smChan_wait(remoteRank);
    }
  } else {
    if (threadIdx.x == 0)
    {
      if (send_end - send_start > 0)
      {
        p.comm_buff.proxyChan_putWithSignal(remoteRank,
                 recv_start * p.cols * sizeof(T),
                 send_start * p.cols * sizeof(T),
                 (send_end - send_start) * p.cols * sizeof(T));
      }
      else
      {
        p.comm_buff.proxyChan_signal(remoteRank);
      }
      p.comm_buff.proxyChan_wait(remoteRank);
    }
  }
}

template<typename T>
__device__ __inline__ size_t max_unidirectional_comm_bytes(All2AllKernelParams<T>& p, int num_tokens_send_to, int remote_rank)
{
  int index = remote_rank * p.num_stages + p.stage_index;
  int begin = index == 0 ? 0 : p.plan.recv_plan[index - 1];
  int end = p.plan.recv_plan[index];
  int num_tokens_recv_from = end - begin;
  return std::max(num_tokens_recv_from, num_tokens_send_to) * p.cols * sizeof(T);
}

template <typename T, int Alignment>
__global__ void All2AllKernel(All2AllKernelParams<T> p) {
  int remoteRank = blockIdx.x;

  int recv_start = p.plan.remote_recv_plan[remoteRank * p.num_stages + p.stage_index];
  int send_start =
      remoteRank == 0 && p.stage_index == 0 ? 0 : p.plan.send_plan[remoteRank * p.num_stages + p.stage_index - 1];
  int send_end = p.plan.send_plan[remoteRank * p.num_stages + p.stage_index];

  size_t bytes = max_unidirectional_comm_bytes<T>(p, send_end - send_start, remoteRank);
  // constexpr size_t use_sm_chan_threshold = 512 * 1024;
  if (is_same_node(p.my_rank, remoteRank, p.num_ranks_per_node)) {
    p.comm_buff.smChan_put(remoteRank,
        recv_start * p.cols * sizeof(T),
        send_start * p.cols * sizeof(T),
        (send_end - send_start) * p.cols * sizeof(T),
        threadIdx.x,
        blockDim.x,
        Alignment);
    __syncthreads();

    if (threadIdx.x == 0) {
      p.comm_buff.smChan_signal(remoteRank);
      p.comm_buff.smChan_wait(remoteRank);
    }
  } else {
    if (threadIdx.x == 0)
    {
      if (send_end - send_start > 0) {
        p.comm_buff.proxyChan_putWithSignal(remoteRank,
          recv_start * p.cols * sizeof(T),
          send_start * p.cols * sizeof(T),
          (send_end - send_start) * p.cols * sizeof(T));
      } else {
        p.comm_buff.proxyChan_signal(remoteRank);
      }
      p.comm_buff.proxyChan_wait(remoteRank);
    }
  }
}

template <typename T, int Alignment>
void All2AllImpl(All2AllKernelParams<T> p, cudaStream_t stream) {
  if (p.stage_index < 0) {
    All2AllOneShotKernel<T, Alignment><<<p.ep_world_size, 512, 0, stream>>>(p);
  } else {
    All2AllKernel<T, Alignment><<<p.ep_world_size, 512, 0, stream>>>(p);
  }
}

template <typename T>
void All2All(All2AllKernelParams<T> p, cudaStream_t stream)
{
  if (p.cols * sizeof(T) % 16 == 0)
  {
    All2AllImpl<T, 16>(p, stream);
  }
  else if (p.cols * sizeof(T) % 8 == 0)
  {
    All2AllImpl<T, 8>(p, stream);
  }
  else if (p.cols * sizeof(T) % 4 == 0)
  {
    All2AllImpl<T, 4>(p, stream);
  } else if (p.cols * sizeof(T) % 2 == 0) {
    All2AllImpl<T, 2>(p, stream);
  } else {
    throw std::runtime_error("Not supported alignment: " + std::to_string(p.cols) + " * " + std::to_string(sizeof(T)));
  }
}

template 
void All2All<int>(All2AllKernelParams<int> p, cudaStream_t stream);

template 
void All2All<float>(All2AllKernelParams<float> p, cudaStream_t stream);

template 
void All2All<half>(All2AllKernelParams<half> p, cudaStream_t stream);


#ifdef ENABLE_BF16
template 
void All2All<__nv_bfloat16>(All2AllKernelParams<__nv_bfloat16> p, cudaStream_t stream);
#endif

template 
void All2All<__nv_fp8_e4m3>(All2AllKernelParams<__nv_fp8_e4m3> p, cudaStream_t stream);


template
void flushProxyChans<int>(FlushProxyChansParams<int> p, cudaStream_t stream);

template
void flushProxyChans<float>(FlushProxyChansParams<float> p, cudaStream_t stream);

template
void flushProxyChans<half>(FlushProxyChansParams<half> p, cudaStream_t stream);

template
void flushProxyChans<__nv_bfloat16>(FlushProxyChansParams<__nv_bfloat16> p, cudaStream_t stream);

template
void flushProxyChans<__nv_fp8_e4m3>(FlushProxyChansParams<__nv_fp8_e4m3> p, cudaStream_t stream);
} // namespace eps