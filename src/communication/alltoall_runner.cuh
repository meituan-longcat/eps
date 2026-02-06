#pragma once

#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>
#include <src/communication/msccl_comm.cuh>

namespace eps {

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;

struct CommPlan {
  int* send_plan; /*[ep_world_size, num_stages]*/
  int* recv_plan; /*[ep_world_size, num_stages]*/
  int* remote_recv_plan; /*[ep_world_size, num_stages]*/
  int* num_will_recv_unique_tokens; /* 1 */
};

struct CommPlanKernelParams {
  CommBuff<int> comm_buff;
  CommPlan plan;
  int* gathered_send_plan; /*[ep_world_size, ep_world_size, num_stages]*/
  int my_rank;
  int num_stages;
  int ep_world_size;
  int num_ranks_per_node;
};

template<typename T>
struct All2AllKernelParams {
  CommBuff<T> comm_buff;
  CommPlan plan;
  int64_t cols;
  int num_stages;
  int stage_index;
  int ep_world_size;
  int my_rank;
  int num_ranks_per_node;
};

void generateCommPlan(CommPlanKernelParams p, cudaStream_t stream);

template <typename T>
void All2All(All2AllKernelParams<T> p, cudaStream_t stream);

template<typename T>
struct FlushProxyChansParams {
  CommBuff<T> comm_buff;
  int my_rank;
  int world_size;
  int num_ranks_per_node;
};

template<typename T>
void flushProxyChans(FlushProxyChansParams<T> p, cudaStream_t stream);

class AlltoallRunner {
 public:
  struct Params {
    int my_rank;
    int num_stages;
    int ep_world_size;
    int num_ranks_per_node;
  };

  AlltoallRunner() = default;

  AlltoallRunner(Params p) : p_{p} {}

  size_t getWorkspaceSize() {
    int recv_plan = p_.ep_world_size * p_.num_stages;
    int remote_recv_plan = p_.ep_world_size * p_.num_stages;

    size_t bytes = (recv_plan + remote_recv_plan) * sizeof(int);
    return bytes;
  }

  CommPlan plan(
      CommBuff<int> comm_buff,
      void* ws,
      int* num_will_recv_unique_tokens,
      cudaStream_t stream
      ) {
    int* recv_plan = (int*)ws;
    int* remote_recv_plan = recv_plan + p_.ep_world_size * p_.num_stages;

    int* send_plan = comm_buff.send_buf;
    int* gathered_send_plan = comm_buff.recv_buf;
    CommPlan plan{.send_plan = send_plan, .recv_plan = recv_plan, .remote_recv_plan = remote_recv_plan, .num_will_recv_unique_tokens = num_will_recv_unique_tokens};

    CommPlanKernelParams p{
        .comm_buff = comm_buff,
        .plan = plan,
        .gathered_send_plan = gathered_send_plan,
        .my_rank = p_.my_rank,
        .num_stages = p_.num_stages,
        .ep_world_size = p_.ep_world_size,
        .num_ranks_per_node = p_.num_ranks_per_node,
    };
    generateCommPlan(p, stream);
    return plan;
  }

  template <typename T>
  void
  run(CommBuff<T> comm_buff, CommPlan comm_plan, int64_t cols, int stage_idx, cudaStream_t stream) {
    All2AllKernelParams<T> p{
        .comm_buff = comm_buff,
        .plan = comm_plan,
        .cols = cols,
        .num_stages = p_.num_stages,
        .stage_index = stage_idx,
        .ep_world_size = p_.ep_world_size,
        .my_rank = p_.my_rank,
        .num_ranks_per_node = p_.num_ranks_per_node,
      };

    All2All<T>(p, stream);
  }

  template <typename T>
  void flush(CommBuff<T> comm_buff, cudaStream_t stream) {
    FlushProxyChansParams<T> p{
      .comm_buff = comm_buff,
      .my_rank = p_.my_rank,
      .world_size = p_.ep_world_size,
      .num_ranks_per_node = p_.num_ranks_per_node,
    };
    flushProxyChans(p, stream);
  }


 private:
  Params p_{};
};

} // namespace eps
