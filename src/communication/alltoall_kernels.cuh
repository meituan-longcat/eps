

#include "src/communication/msccl_comm.cuh"

namespace eps {

template<typename T>
struct All2AllParams {
    int64_t my_rank;
    int64_t remote_rank;
    int64_t num_ranks_per_node;
    int64_t world_size;

    CommBuff<T> comm_buff;

    int64_t send_start;
    int64_t send_end;
    int64_t recv_start;
    int64_t cols;
};

template <typename T, int Alignment>
__device__ void All2AllKernelImplSameNode(All2AllParams<T> p) {
    p.comm_buff.smChan_put(p.remote_rank,
        p.recv_start * p.cols * sizeof(T),
        p.send_start * p.cols * sizeof(T),
        (p.send_end - p.send_start) * p.cols * sizeof(T),
        threadIdx.x,
        blockDim.x,
        Alignment);
    __syncthreads();

    if (threadIdx.x == 0) {
      p.comm_buff.smChan_signal(p.remote_rank);
      p.comm_buff.smChan_wait(p.remote_rank);
    }
}

template <typename T, int Alignment>
__device__ void All2AllKernelImplCrossNode(All2AllParams<T> p) {
    if (p.send_end - p.send_start > 0)
    {
      p.comm_buff.proxyChan_putWithSignal(p.remote_rank,
            p.recv_start * p.cols * sizeof(T),
            p.send_start * p.cols * sizeof(T),
            (p.send_end - p.send_start) * p.cols * sizeof(T));
    }
    else
    {
      p.comm_buff.proxyChan_signal(p.remote_rank);
    }
    p.comm_buff.proxyChan_wait(p.remote_rank);
}


template <typename T, int Alignment, template <typename> class ParamTemplate>
__global__ void All2AllKernel(ParamTemplate<T> p) {
  int my_node = p.my_rank / p.num_ranks_per_node;
  int my_node_rank_start = my_node * p.num_ranks_per_node;
  int my_node_rank_end = (my_node + 1) * p.num_ranks_per_node;

  if (blockIdx.x < p.num_ranks_per_node) {
    int remote_rank = my_node_rank_start + (p.my_rank - my_node_rank_start + blockIdx.x) % p.num_ranks_per_node;
    All2AllParams<T> params = p.prepare(remote_rank);
    All2AllKernelImplSameNode<T, Alignment>(params);
  } else {
    if (threadIdx.x < (p.world_size - p.num_ranks_per_node)) {
      int remote_rank = (my_node_rank_end + threadIdx.x) % p.world_size;
      All2AllParams<T> params = p.prepare(remote_rank);
      All2AllKernelImplCrossNode<T, Alignment>(params);
    }
  }
}

template <typename T, typename Params>
void All2All(Params p, cudaStream_t stream)
{
  int num_blocks = p.num_ranks_per_node + 1;
  int block_size = 512;
  if (p.cols * sizeof(T) % 16 == 0)
  {
    All2AllKernel<T, 16><<<num_blocks, block_size, 0, stream>>>(p);
  }
  else if (p.cols * sizeof(T) % 8 == 0)
  {
    All2AllKernel<T, 8><<<num_blocks, block_size, 0, stream>>>(p);
  }
  else if (p.cols * sizeof(T) % 4 == 0)
  {
    All2AllKernel<T, 4><<<num_blocks, block_size, 0, stream>>>(p);
  } else if (p.cols * sizeof(T) % 2 == 0)
  {
    All2AllKernel<T, 2><<<num_blocks, block_size, 0, stream>>>(p);
  } else {
    throw std::runtime_error("Not supported alignment: " + std::to_string(p.cols) + " * " + std::to_string(sizeof(T)));
  }
}

}
