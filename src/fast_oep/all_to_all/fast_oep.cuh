#pragma once

#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include "src/scheduler/gemm_executor_interface.cuh"
#include "src/scheduler/common.cuh"
#include "src/communication/msccl_comm.cuh"
#include "src/fast_oep/all_to_all/fast_oep_kernels.cuh"
#include "src/common/debug.cuh"

namespace fast_oep {

template <typename DispatchType, typename CombineType>
struct CommBuffs
{
    struct Params
    {
        int64_t num_global_tokens;
        int64_t embed_dim;
        MscclppCommunicator *comm;

        std::string str() {
            std::ostringstream oss;
            oss << "Params(num_global_tokens: " << num_global_tokens << " embed_dim: " << embed_dim << ")";
            return oss.str();
        }
    };

    CommBuffs(Params p);

    Params p_{};
    CommBuffHost<DispatchType> input;
    CommBuffHost<CombineType> output;
    CommBuffHost<int> comm_plan;

    private:
        using TArrPtr = std::unique_ptr<char[], mscclpp::CudaDeleter<char[]>>;

        std::vector<CommBuffBase*> buffs_;

        char *send_buf;
        char *recv_buf;
        TArrPtr send_buf_;
        TArrPtr recv_buf_;
        std::vector<mscclpp::SmChannel> smChans_;
        HandleArrPtr smChansHandle_;
        std::vector<mscclpp::DeviceHandle<mscclpp::ProxyChannel>> proxyChans_;
        ProxyHandleArrPtr proxyChansHandle_;

        Handle *smChans;
        ProxyHandle *proxyChans;
        MscclppCommunicator::SemaphoreMap smSemaphores_;
};

template<typename T>
class FastOEP {
public:
    struct Params {
        int64_t embed_dim;
        int64_t size_per_rank;
        int64_t max_num_global_tokens;
        MscclppCommunicator *comm;
    };

    struct WorkspaceOffsets {
        struct AllGatheredPlan {
            int64_t remote_recv_plan;
            int64_t local_send_plan;
        } all_gathered_plan;

        struct ArrangePlan {
            int64_t mapping;
        } arrange_plan;

        int64_t total_bytes;

        WorkspaceOffsets(Params p);
    };

public:
  FastOEP(Params p);

  void dispatch(
    const int32_t *ids,
    int64_t num_tokens,
    cudaStream_t stream);

  void combine(
    T* out,
    int64_t num_tokens,
    int64_t num_global_tokens,
    int64_t n_grams,
    bool do_permute,
    const T* embed_table,
    cudaStream_t stream);

private:
  void flush(cudaStream_t stream);

  Params p_;
  WorkspaceOffsets workspace_offsets_;
  char *ws_{};
  std::unique_ptr<CommBuffs<int32_t, T>> comm_buffs_;
};

}
