#include "src/fast_ep/all_to_all/fast_ep.cuh"

#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "src/common/cuda_utils.cuh"
#include "src/common/common.cuh"
#include "src/scheduler/utils.cuh"
#include "src/communication/alltoall_runner.cuh"

// #include <cuda_profiler_api.h>

namespace fast_ep {

template<typename T>
FastEP<T>::FastEP(Params p): p_{p}, workspace_offsets_(p) {
    cudaStreamCreate(&side_stream_);
    cudaEventCreate(&event_);
    p_.comm->barrier();
    comm_buffs_ = std::make_unique<CommBuffs<T, T>>(
        typename CommBuffs<T, T>::Params{
            .topk = (int)p_.topk,
            .num_experts = p_.num_experts,
            .num_global_tokens = p_.max_num_global_tokens,
            .hidden_size = p_.hidden_size,
            .comm = p_.comm
        }
    );
    cudaDeviceSynchronize();
    p_.comm->barrier();

    cudaMalloc(&ws_, workspace_offsets_.total_bytes);
}

template<typename T>
FastEP<T>::WorkspaceOffsets::WorkspaceOffsets(Params p) {
    int64_t bytes = 0;

    expand_plan.mapping = bytes;
    bytes += p.max_num_global_tokens * p.topk * sizeof(int);

    all_gathered_plan.remote_recv_plan = bytes;
    bytes += p.comm->getWorldSize() * sizeof(int);
    all_gathered_plan.local_send_plan = bytes;
    bytes += (p.comm->getWorldSize() + 1) * sizeof(int);

    arrange_plan.src = bytes;
    bytes += p.num_experts * sizeof(int);
    arrange_plan.rows = bytes;
    bytes += p.num_experts * sizeof(int);
    arrange_plan.dst = bytes;
    bytes += p.num_experts * sizeof(int);

    bytes = ALIGN(bytes);

    total_bytes = bytes;
}

template<typename T>
void FastEP<T>::dispatch(
    int32_t *exclusive_sum,
    T* expertX,
    T* dpX,
    const int32_t *expert_indices,
    int32_t num_tokens,
    int32_t num_global_tokens,
    cudaStream_t stream) {
    sync_check_cuda_error_eps();
    // cudaProfilerStart();
    context_.dpX = dpX;

    ExpandPlan expand_plan{
        .num_tokens = num_tokens,
        .topk = p_.topk,
        .num_experts = p_.num_experts,
        .expert_indices = expert_indices,
        .exclusive_sum = comm_buffs_->comm_plan.send_buf,
        .mapping = (int*)(ws_ + workspace_offsets_.expand_plan.mapping)
    };
    context_.expand_plan = expand_plan;
    expandPlan(expand_plan, stream);
    sync_check_cuda_error_eps();

    cudaEventRecord(event_, stream);
    expand(ExpandParams<T>{
        .src = dpX,
        .dst = comm_buffs_->input_hidden.send_buf,
        .cols = p_.hidden_size,
        .plan = expand_plan
    }, stream);
    sync_check_cuda_error_eps();

    ArrangePlan arrange_plan{
        .src = (int *)(ws_ + workspace_offsets_.arrange_plan.src),
        .rows = (int *)(ws_ + workspace_offsets_.arrange_plan.rows),
        .dst = (int *)(ws_ + workspace_offsets_.arrange_plan.dst),
    };
    context_.arrange_plan = arrange_plan;

    AllGatheredPlan plan{
        .my_rank = p_.comm->getRank(),
        .num_ranks_per_node = p_.comm->getNumRanksPerNode(),
        .ep_world_size = p_.comm->getWorldSize(),
        .num_experts = p_.num_experts,
        .all_gathered = comm_buffs_->comm_plan.recv_buf,
        .remote_recv_plan = (int*)(ws_ + workspace_offsets_.all_gathered_plan.remote_recv_plan),
        .local_send_plan = (int*)(ws_ + workspace_offsets_.all_gathered_plan.local_send_plan),
        .gemm_plan = exclusive_sum,
        .arrange_plan = arrange_plan,
        .comm_buff = comm_buffs_->comm_plan,
    };


    {
        cudaStreamWaitEvent(side_stream_, event_);
        allGatherPlan(plan, side_stream_);
        sync_check_cuda_error_eps();
        cudaEventRecord(event_, side_stream_);
    }
    cudaStreamWaitEvent(stream, event_);

    All2All<T>(
        DispatchAll2AllParams<T>{
            .my_rank = p_.comm->getRank(),
            .num_ranks_per_node = p_.comm->getNumRanksPerNode(),
            .world_size = p_.comm->getWorldSize(),
            .num_experts = p_.num_experts,
            .cols = p_.hidden_size,
            .exclusive_sum = comm_buffs_->comm_plan.send_buf,
            .remote_recv_plan = (int*)(ws_ + workspace_offsets_.all_gathered_plan.remote_recv_plan),
            .comm_buff = comm_buffs_->input_hidden,
        },
        stream
    );
    sync_check_cuda_error_eps();

    arrange(
        ArrangeParams<T>{
            .ep_world_size = p_.comm->getWorldSize(),
            .num_experts = p_.num_experts,
            .cols = p_.hidden_size,
            .src = comm_buffs_->input_hidden.recv_buf,
            .dst = expertX,
            .plan = arrange_plan,
            .num_tokens_hint = CEILDIV(num_global_tokens * p_.topk, p_.comm->getWorldSize())
        },
        stream
    );
    sync_check_cuda_error_eps();
}

template<typename T>
void FastEP<T>::combine(
    T* out_tokens,
    const float* expert_scales,
    const T* expertX,
    int32_t num_global_tokens,
    cudaStream_t stream) {
    arrange(
        ArrangeParams<T>{
            .ep_world_size = p_.comm->getWorldSize(),
            .num_experts = p_.num_experts,
            .cols = p_.hidden_size,
            .src = expertX,
            .dst = comm_buffs_->output_hidden.send_buf,
            .plan = ArrangePlan{
                .src = context_.arrange_plan.dst,
                .rows = context_.arrange_plan.rows,
                .dst = context_.arrange_plan.src
                },
            .num_tokens_hint = CEILDIV(num_global_tokens * p_.topk, p_.comm->getWorldSize())
        },
        stream
    );
    sync_check_cuda_error_eps();

    All2All<T>(
        CombineAll2AllParams<T>{
            .my_rank = p_.comm->getRank(),
            .num_ranks_per_node = p_.comm->getNumRanksPerNode(),
            .world_size = p_.comm->getWorldSize(),
            .num_experts = p_.num_experts,
            .cols = p_.hidden_size,
            .all_gathered = comm_buffs_->comm_plan.recv_buf,
            .local_send_plan = (int*)(ws_ + workspace_offsets_.all_gathered_plan.local_send_plan),
            .comm_buff = comm_buffs_->output_hidden,
        },
        stream
    );
    sync_check_cuda_error_eps();

    reduce(
        ReduceParams<T, float>{
            .src = comm_buffs_->output_hidden.recv_buf,
            .dst = out_tokens,
            .shared = nullptr,
            .raw_tokens = context_.dpX,
            .expert_scales = expert_scales,
            .cols = p_.hidden_size,
            .plan = context_.expand_plan
        },
        stream
    );
    sync_check_cuda_error_eps();

    const int num_nodes = p_.comm->getWorldSize() / p_.comm->getNumRanksPerNode();
    if (num_nodes > 1) {
        flush(stream);
    }

    // cudaProfilerStop();
}

template<typename T>
void FastEP<T>::flush(cudaStream_t stream) {
    FlushProxyChansParams<int> p{
        .comm_buff = comm_buffs_->comm_plan,
        .my_rank = (int)p_.comm->getRank(),
        .world_size = (int)p_.comm->getWorldSize(),
        .num_ranks_per_node = (int)p_.comm->getNumRanksPerNode(),
    };
    flushProxyChans(p, stream);
}

template
class FastEP<float>;

template
class FastEP<half>;

template
class FastEP<__nv_bfloat16>;

template <typename DispatchType, typename CombineType>
CommBuffs<DispatchType, CombineType>::CommBuffs(Params p) :
    p_{p},
    input_hidden{p.num_global_tokens * p.topk * p.hidden_size, p.num_global_tokens * p.topk * p.hidden_size, p.comm->getRank(), p.comm->getNumRanksPerNode()},
    output_hidden{p.num_global_tokens * p.topk * p.hidden_size, p.num_global_tokens * p.topk * p.hidden_size, p.comm->getRank(), p.comm->getNumRanksPerNode()},
    comm_plan{p.num_experts + 1, p.comm->getWorldSize() * (p.num_experts + 1), p.comm->getRank(), p.comm->getNumRanksPerNode()}
{
    buffs_.push_back(&input_hidden);
    buffs_.push_back(&output_hidden);
    buffs_.push_back(&comm_plan);

    int64_t send_bytes = 0;
    int64_t recv_bytes = 0;
    for (CommBuffBase* buff : buffs_) {
        buff->set_send_offset(send_bytes);
        send_bytes += buff->send_bytes;

        buff->set_recv_offset(recv_bytes);
        recv_bytes += buff->recv_bytes;
    }

    if (p.comm->isCrossNode()) {
        if (send_bytes >= (1LL << 32) || recv_bytes >= (1LL << 32)) {
            std::string err = "send_bytes: " + std::to_string(send_bytes) + ", recv_bytes: " + std::to_string(recv_bytes) + " is too big.";
            err = err + " " + p.str();
            throw std::runtime_error(err);
        }
    }

    if (p.comm->getRank() == 0) {
        fprintf(stderr, "\033[32m[[Global Rank%d] fast ep param: %s, send_bytes: %ld, recv_bytes: %ld]\033[0m\n", p.comm->getRank(), p.str().c_str(), send_bytes, recv_bytes);
    }

    if (send_bytes) {
        cudaMalloc(&send_buf, send_bytes);
        send_buf_ = TArrPtr{send_buf};
    }

    if (recv_bytes) {
        cudaMalloc(&recv_buf, recv_bytes);
        recv_buf_ = TArrPtr{recv_buf};
    }

    cudaMalloc(&smChans, sizeof(Handle) * p.comm->getWorldSize() * NUM_CHANNELS_PER_CONNECTION);
    smChansHandle_ = HandleArrPtr{smChans};

    std::tie(smChans_, proxyChans_, smSemaphores_) = p.comm->createChans(send_buf, send_bytes, recv_buf, recv_bytes);
    MscclppCommunicator::getSmChanDeviceHandle(smChans, smChans_);

    cudaMalloc(&proxyChans, sizeof(ProxyHandle) * p.comm->getWorldSize());
    cudaMemcpy(proxyChans, proxyChans_.data(), sizeof(ProxyHandle) * p.comm->getWorldSize(), cudaMemcpyHostToDevice);
    proxyChansHandle_ = ProxyHandleArrPtr{proxyChans};

    for (CommBuffBase* buff : buffs_) {
        buff->set_recv(recv_buf);
        buff->set_send(send_buf);
        buff->set_smChans(smChans);
        buff->set_proxyChans(proxyChans);
    }
}

}
