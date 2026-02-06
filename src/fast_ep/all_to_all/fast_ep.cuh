#pragma once

#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include "src/scheduler/gemm_executor_interface.cuh"
#include "src/scheduler/common.cuh"
#include "src/communication/msccl_comm.cuh"
#include "src/fast_ep/all_to_all/fast_ep_kernels.cuh"
#include "src/common/debug.cuh"

namespace fast_ep {

template <typename DispatchType, typename CombineType>
struct CommBuffs
{
    struct Params
    {
        int topk;
        int64_t num_experts;
        int64_t num_global_tokens;
        int64_t hidden_size;
        MscclppCommunicator *comm;

        std::string str() {
            std::ostringstream oss;
            oss << "Params(topk: " << topk << " num_experts: " << num_experts << " num_global_tokens: " << num_global_tokens << " hidden_size: " << hidden_size << ")";
            return oss.str();
        }
    };

    CommBuffs(Params p);

    Params p_{};
    CommBuffHost<DispatchType> input_hidden;
    CommBuffHost<CombineType> output_hidden;
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
class FastEP {
public:
    struct Params {
        int64_t topk;
        int64_t num_experts;
        int64_t hidden_size;
        int64_t max_num_global_tokens;
        MscclppCommunicator *comm;
    };

    struct WorkspaceOffsets {
        struct ExpandPlan {
            int64_t mapping;
        } expand_plan;

        struct AllGatheredPlan {
            int64_t remote_recv_plan;
            int64_t local_send_plan;
        } all_gathered_plan;

        struct ArrangePlan {
            int64_t src;
            int64_t rows;
            int64_t dst;
        } arrange_plan;

        int64_t total_bytes;

        WorkspaceOffsets(Params p);
    };

    struct Context {
        ArrangePlan arrange_plan;
        ExpandPlan expand_plan;
        T* dpX;
    };

public:
  FastEP(Params p);

  void dispatch(int32_t *exclusive_sum,
                T *expertX,
                T *dpX,
                const int32_t *expert_indices,
                int32_t num_tokens,
                int32_t num_global_tokens,
                cudaStream_t stream);

  void combine(T *out_tokens,
               const float *expert_scales,
               const T *expertX,
               int32_t num_global_tokens,
               cudaStream_t stream);

private:
  void flush(cudaStream_t stream);

  Params p_;
  WorkspaceOffsets workspace_offsets_;
  char *ws_{};
  std::unique_ptr<CommBuffs<T, T>> comm_buffs_;

  Context context_;
  cudaStream_t side_stream_;
  cudaEvent_t event_;
};

}