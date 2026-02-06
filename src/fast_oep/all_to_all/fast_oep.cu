#include "src/fast_oep/all_to_all/fast_oep.cuh"

#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "src/common/cuda_utils.cuh"
#include "src/common/common.cuh"
#include "src/scheduler/utils.cuh"
#include "src/communication/alltoall_runner.cuh"

// #include <cuda_profiler_api.h>

namespace fast_oep {

template<typename T>
FastOEP<T>::FastOEP(Params p): p_{p}, workspace_offsets_(p) {
    p_.comm->barrier();
    comm_buffs_ = std::make_unique<CommBuffs<int32_t, T>>(
        typename CommBuffs<int32_t, T>::Params{
            .num_global_tokens = p_.max_num_global_tokens,
            .embed_dim = p_.embed_dim,
            .comm = p_.comm
        }
    );
    cudaDeviceSynchronize();
    p_.comm->barrier();

    cudaMalloc(&ws_, workspace_offsets_.total_bytes);
}

template<typename T>
FastOEP<T>::WorkspaceOffsets::WorkspaceOffsets(Params p) {
    int64_t bytes = 0;

    all_gathered_plan.remote_recv_plan = bytes;
    bytes += p.comm->getWorldSize() * sizeof(int);
    all_gathered_plan.local_send_plan = bytes;
    bytes += (p.comm->getWorldSize() + 1) * sizeof(int);

    arrange_plan.mapping = bytes;
    bytes += p.max_num_global_tokens * sizeof(int);

    bytes = ALIGN(bytes);

    total_bytes = bytes;
}

template<typename T>
void FastOEP<T>::dispatch(
    const int32_t *ids,
    int64_t num_tokens,
    cudaStream_t stream) {
    sync_check_cuda_error_eps();
    ArrangePlan arrange_plan{
        .size_per_rank = p_.size_per_rank,
        .num_tokens = num_tokens,
        .world_size = p_.comm->getWorldSize(),
        .ids = ids,
        .arranged_ids = comm_buffs_->input.send_buf,
        .exclusive_sum = comm_buffs_->comm_plan.send_buf,
        .mapping = (int*)(ws_ + workspace_offsets_.arrange_plan.mapping)
    };
    arrangePlan(arrange_plan, stream);

    sync_check_cuda_error_eps();
    AllGatheredPlan plan{
        .my_rank = p_.comm->getRank(),
        .num_ranks_per_node = p_.comm->getNumRanksPerNode(),
        .world_size = p_.comm->getWorldSize(),
        .all_gathered = comm_buffs_->comm_plan.recv_buf,
        .remote_recv_plan = (int*)(ws_ + workspace_offsets_.all_gathered_plan.remote_recv_plan),
        .local_send_plan = (int*)(ws_ + workspace_offsets_.all_gathered_plan.local_send_plan),
        .comm_buff = comm_buffs_->comm_plan,
    };
    allGatherPlan(plan, stream);
    sync_check_cuda_error_eps();
    All2All<int32_t>(
        DispatchAll2AllParams<int32_t>{
            .my_rank = p_.comm->getRank(),
            .num_ranks_per_node = p_.comm->getNumRanksPerNode(),
            .world_size = p_.comm->getWorldSize(),
            .exclusive_sum = comm_buffs_->comm_plan.send_buf,
            .remote_recv_plan = (int*)(ws_ + workspace_offsets_.all_gathered_plan.remote_recv_plan),
            .comm_buff = comm_buffs_->input,
        },
        stream
    );
    sync_check_cuda_error_eps();
}

template<typename T>
void FastOEP<T>::combine(
    T* out,
    int64_t num_tokens,
    int64_t num_global_tokens,
    int64_t n_grams,
    bool do_permute,
    const T* embed_table,
    cudaStream_t stream) {
    sync_check_cuda_error_eps();
    lookup(
        LookupParams<T>{
            .size_per_rank = p_.size_per_rank,
            .world_size = p_.comm->getWorldSize(),
            .num_tokens_hint = CEILDIV(num_global_tokens, p_.comm->getWorldSize()),
            .embed_dim = p_.embed_dim,
            .ids = comm_buffs_->input.recv_buf,
            .local_send_plan = (int*)(ws_ + workspace_offsets_.all_gathered_plan.local_send_plan),
            .embed_table = embed_table,
            .dst = comm_buffs_->output.send_buf
        },
        stream
    );
    sync_check_cuda_error_eps();

    All2All<T>(
        CombineAll2AllParams<T>{
            .my_rank = p_.comm->getRank(),
            .num_ranks_per_node = p_.comm->getNumRanksPerNode(),
            .world_size = p_.comm->getWorldSize(),
            .cols = p_.embed_dim,
            .all_gathered = comm_buffs_->comm_plan.recv_buf,
            .local_send_plan = (int*)(ws_ + workspace_offsets_.all_gathered_plan.local_send_plan),
            .comm_buff = comm_buffs_->output,
        },
        stream
    );
    sync_check_cuda_error_eps();

    scatter(
        ScatterParams<T>{
            .mapping = (int*)(ws_ + workspace_offsets_.arrange_plan.mapping),
            .num_tokens = num_tokens,
            .embed_dim = p_.embed_dim,
            .n_grams = n_grams,
            .do_permute = do_permute,
            .src = comm_buffs_->output.recv_buf,
            .dst = out
        },
        stream
    );
    sync_check_cuda_error_eps();

    const int num_nodes = p_.comm->getWorldSize() / p_.comm->getNumRanksPerNode();
    if (num_nodes > 1) {
        flush(stream);
    }
}

template<typename T>
void FastOEP<T>::flush(cudaStream_t stream) {
    FlushProxyChansParams<int> p{
        .comm_buff = comm_buffs_->comm_plan,
        .my_rank = (int)p_.comm->getRank(),
        .world_size = (int)p_.comm->getWorldSize(),
        .num_ranks_per_node = (int)p_.comm->getNumRanksPerNode(),
    };
    flushProxyChans(p, stream);
}

template
class FastOEP<float>;

template
class FastOEP<half>;

template
class FastOEP<__nv_bfloat16>;

template <typename DispatchType, typename CombineType>
CommBuffs<DispatchType, CombineType>::CommBuffs(Params p) :
    p_{p},
    input{p.num_global_tokens, p.num_global_tokens, p.comm->getRank(), p.comm->getNumRanksPerNode()},
    output{p.num_global_tokens * p.embed_dim, p.num_global_tokens * p.embed_dim, p.comm->getRank(), p.comm->getNumRanksPerNode()},
    comm_plan{p.comm->getWorldSize() + 1, p.comm->getWorldSize() * (p.comm->getWorldSize() + 1), p.comm->getRank(), p.comm->getNumRanksPerNode()}
{
    buffs_.push_back(&input);
    buffs_.push_back(&output);
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
        fprintf(stderr, "\033[32m[[Global Rank%d] fast oep param: %s, send_bytes: %ld, recv_bytes: %ld]\033[0m\n", p.comm->getRank(), p.str().c_str(), send_bytes, recv_bytes);
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
