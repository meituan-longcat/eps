#include "src/communication/msccl_comm.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "src/common/common.cuh"

namespace eps
{

    void MscclppCommunicator::enableP2P(int ngpus)
    {
        int peer_access_available = 0;
        for (int i = 0; i < ngpus; i++)
        {
            cudaSetDevice(i);
            for (int j = 0; j < ngpus; j++)
            {
                if (i == j)
                {
                    continue;
                }
                cudaDeviceCanAccessPeer(&peer_access_available, i, j);
                // Custom AR Kernels need DGX A100 NVSWITCH connections
                assert(peer_access_available);
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    void MscclppCommunicator::allGather(void* allData, int size) {
        comm_->bootstrap()->allGather(allData, size);
    }

    void MscclppCommunicator::broadcast(std::vector<char>& data) {
        int tag = 0;
        if (p_.rank == 0) {
            for (int peer = 1; peer < p_.ep_world_size; peer++) {
                comm_->bootstrap()->send(data, peer, tag);
            }
        } else {
            comm_->bootstrap()->recv(data, 0, tag);
        }
    }

    MscclppCommunicator::MscclppCommunicator(mscclpp::UniqueId id, Params p) : p_{p}
    {
        // std::string ibDevStr = "mlx5_" + std::to_string(p_.rank % p_.num_ranks_per_node);
        // ibTransport_ = mscclpp::getIBTransportByDeviceName(ibDevStr);
        if (p_.ep_world_size == p_.num_ranks_per_node) {
            ibTransport_ = mscclpp::Transport::CudaIpc;
        } else {
            ibTransport_ = mscclpp::getIBTransportByLocalRank(p_.rank % p_.num_ranks_per_node);
        }

        auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(p_.rank, p_.ep_world_size);
        constexpr int64_t timeout = 1000000000;
        bootstrap->initialize(id, timeout);
        comm_ = std::make_unique<mscclpp::Communicator>(bootstrap);

        barrier();
        setupConnections();
        setupProxyService();
        barrier();
    }


    void MscclppCommunicator::setupProxyService() {
        chanService_ = std::make_shared<mscclpp::ProxyService>(102400);
        chanService_->startProxy();
    }

    void MscclppCommunicator::setupConnections()
    {
        auto rankToNode = [num_ranks_per_node=p_.num_ranks_per_node](int rank) { return rank / num_ranks_per_node; };

        std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
        for (int r = 0; r < p_.ep_world_size; r++)
        {
            if (rankToNode(r) != rankToNode(p_.rank)) {
                connectionFutures.push_back(comm_->connectOnSetup(r, 0, ibTransport_));
            } else {
                connectionFutures.push_back(comm_->connectOnSetup(r, 0, mscclpp::Transport::CudaIpc));
            }
        }
        comm_->setup();

        std::transform(
            connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections_),
            [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>> &future)
            { return future.get(); });
    }

    MscclppCommunicator::SemaphoreMap MscclppCommunicator::setupSemaphores() {
        SemaphoreMap smSemaphores;
        for (size_t connect_id = 0; connect_id < connections_.size(); ++connect_id)
        {
            for (size_t channel_id = 0; channel_id < NUM_CHANNELS_PER_CONNECTION; ++channel_id)
            {
                smSemaphores[connect_id].emplace_back(std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm_, connections_[connect_id]));

            }
        }
        comm_->setup();

        return smSemaphores;
    }

    std::tuple<std::vector<mscclpp::SmChannel>, std::vector<mscclpp::DeviceHandle<mscclpp::ProxyChannel>>, MscclppCommunicator::SemaphoreMap>
    MscclppCommunicator::createChans(void *inputBuf, size_t inputBufBytes, void *outputBuf, size_t outputBufBytes)
    {
        mscclpp::RegisteredMemory inputBufRegMem = comm_->registerMemory(inputBuf, inputBufBytes, mscclpp::Transport::CudaIpc | ibTransport_);
        mscclpp::RegisteredMemory outputBufRegMem;
        if (outputBufBytes)
        {
            outputBufRegMem = comm_->registerMemory(outputBuf, outputBufBytes, mscclpp::Transport::CudaIpc | ibTransport_);
        }

        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
        for (int r = 0; r < p_.ep_world_size; r++)
        {
            comm_->sendMemoryOnSetup(outputBufBytes ? outputBufRegMem : inputBufRegMem, r, 0);
            auto remoteMemory = comm_->recvMemoryOnSetup(r, 0);
            remoteRegMemories.push_back(remoteMemory);
        }
        comm_->setup();

        SemaphoreMap smSemaphores = setupSemaphores();

        std::vector<mscclpp::SmChannel> smChans;

        for (size_t channel_id = 0; channel_id < NUM_CHANNELS_PER_CONNECTION; ++channel_id)
        {
            for (size_t connect_id = 0; connect_id < connections_.size(); ++connect_id)
            {
                smChans.emplace_back(smSemaphores[connect_id][channel_id], remoteRegMemories[connect_id].get(),
                                     inputBufRegMem.data(),
                                     outputBuf);
            }
        }

        std::vector<mscclpp::DeviceHandle<mscclpp::ProxyChannel>> proxyChannels;
        auto service = std::dynamic_pointer_cast<mscclpp::ProxyService>(chanService_);
        for (size_t i = 0; i < connections_.size(); ++i) {
            proxyChannels.push_back(mscclpp::deviceHandle(
                service->proxyChannel(
                    service->buildAndAddSemaphore(*comm_, connections_[i]),
                    service->addMemory(remoteRegMemories[i].get()),
                    service->addMemory(inputBufRegMem)
                )
                )
            );
        }

        comm_->setup();
        return {smChans, proxyChannels, smSemaphores};
    }

    void MscclppCommunicator::getSmChanDeviceHandle(void *d_ptr, std::vector<mscclpp::SmChannel> &smChans)
    {
        auto getChannelDeviceHandle = [](const std::vector<mscclpp::SmChannel> &in,
                                         std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> &out)
        {
            return std::transform(in.begin(), in.end(), out.begin(),
                                  [](const mscclpp::SmChannel &smChannel)
                                  { return mscclpp::deviceHandle(smChannel); });
        };
        std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles(smChans.size());
        getChannelDeviceHandle(smChans, smChannelDeviceHandles);
        cudaMemcpy(d_ptr, smChannelDeviceHandles.data(), sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>) * smChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
    }

    std::shared_ptr<std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>>>
    MscclppCommunicator::getSmChanDeviceHandleAsync(void *d_ptr, std::vector<mscclpp::SmChannel> &smChans, cudaStream_t stream)
    {
        auto getChannelDeviceHandle = [](const std::vector<mscclpp::SmChannel> &in,
                                         std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> &out)
        {
            return std::transform(in.begin(), in.end(), out.begin(),
                                  [](const mscclpp::SmChannel &smChannel)
                                  { return mscclpp::deviceHandle(smChannel); });
        };
        auto smChannelDeviceHandlesPtr = std::make_shared<std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>>>(smChans.size());
        getChannelDeviceHandle(smChans, *smChannelDeviceHandlesPtr);
        cudaMemcpyAsync(d_ptr, smChannelDeviceHandlesPtr->data(), sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>) * smChannelDeviceHandlesPtr->size(), cudaMemcpyHostToDevice, stream);
        return smChannelDeviceHandlesPtr;
    }

    template <typename DispatchType, typename CombineType, typename ExpertScaleType>
    CommBuffs<DispatchType, CombineType, ExpertScaleType>::CommBuffs(Params p) :
            expert_scales{p.max_num_tokens_per_gpu * p.ep_world_size * p.topk, p.max_num_tokens_per_gpu * p.ep_world_size * p.topk, p.comm->getRank(), p.comm->getNumRanksPerNode()},
            expert_indices{p.max_num_tokens_per_gpu * p.ep_world_size * p.topk, p.max_num_tokens_per_gpu * p.ep_world_size * p.topk, p.comm->getRank(), p.comm->getNumRanksPerNode()},
            input_hidden{p.max_num_tokens_per_gpu * p.ep_world_size * p.hidden_size, p.max_num_tokens_per_gpu * p.ep_world_size * p.hidden_size, p.comm->getRank(), p.comm->getNumRanksPerNode()},
            output_hidden{p.max_num_tokens_per_gpu * p.ep_world_size * p.hidden_size, p.max_num_tokens_per_gpu * p.ep_world_size * p.hidden_size, p.comm->getRank(), p.comm->getNumRanksPerNode()},
            dispatch_comm_plan{p.ep_world_size * p.max_num_stages, p.ep_world_size * p.ep_world_size * p.max_num_stages, p.comm->getRank(), p.comm->getNumRanksPerNode()},
            combine_comm_plan{p.ep_world_size * p.max_num_stages, p.ep_world_size * p.ep_world_size * p.max_num_stages, p.comm->getRank(), p.comm->getNumRanksPerNode()}
    {
        buffs_.push_back(&expert_scales);
        buffs_.push_back(&expert_indices);
        buffs_.push_back(&input_hidden);
        buffs_.push_back(&output_hidden);
        buffs_.push_back(&dispatch_comm_plan);
        buffs_.push_back(&combine_comm_plan);

        {
            input_scales = CommBuffHost<float>(p.max_num_tokens_per_gpu * p.ep_world_size * p.input_scales_dim, p.max_num_tokens_per_gpu * p.ep_world_size * p.input_scales_dim, p.comm->getRank(), p.comm->getNumRanksPerNode());
            buffs_.push_back(&input_scales);
        }

        int64_t send_bytes = 0;
        int64_t recv_bytes = 0;
        for (CommBuffBase* buff : buffs_) {
            buff->set_send_offset(send_bytes);
            send_bytes += ALIGN(buff->send_bytes);

            buff->set_recv_offset(recv_bytes);
            recv_bytes += ALIGN(buff->recv_bytes);
        }

        if (p.comm->isCrossNode()) {
            if (send_bytes >= (1LL << 32) || recv_bytes >= (1LL << 32)) {
                std::string err = "send_bytes: " + std::to_string(send_bytes) + ", recv_bytes: " + std::to_string(recv_bytes) + " is too big.";
                err = err + " " + p.str();
                throw std::runtime_error(err);
            }
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

    template
    struct CommBuff<float>;

    template
    struct CommBuff<__half>;

    template
    struct CommBuff<__nv_bfloat16>;

    template
    struct CommBuff<__nv_fp8_e4m3>;

    template
    struct CommBuff<int>;

    template
    struct CommBuffs<float, float, float>;

    template
    struct CommBuffs<__half, __half, __half>;
    template
    struct CommBuffs<__half, __half, float>;

    template
    struct CommBuffs<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
    template
    struct CommBuffs<__nv_bfloat16, __nv_bfloat16, float>;

    template
    struct CommBuffs<__nv_fp8_e4m3, float, float>;
    template
    struct CommBuffs<__nv_fp8_e4m3, __half, __half>;
    template
    struct CommBuffs<__nv_fp8_e4m3, __half, float>;

    template
    struct CommBuffs<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16>;
    template
    struct CommBuffs<__nv_fp8_e4m3, __nv_bfloat16, float>;
}
