#pragma once

#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <sstream>

const size_t NUM_CHANNELS_PER_CONNECTION = 64;

namespace eps
{

    class MscclppCommunicator
    {
    public:
        using SemaphoreMap = std::unordered_map<size_t, std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>>;

    public:
        static void enableP2P(int ngpus);

        static mscclpp::UniqueId createUniqueId()
        {
            return mscclpp::TcpBootstrap::createUniqueId();
        }

        struct Params
        {
            int rank;
            int ep_world_size;
            int num_ranks_per_node;
        };

        MscclppCommunicator(mscclpp::UniqueId id, Params p);

        void barrier() { this->comm_->bootstrap()->barrier(); }

        std::tuple<std::vector<mscclpp::SmChannel>,
                   std::vector<mscclpp::DeviceHandle<mscclpp::ProxyChannel>>,
                   SemaphoreMap>
        createChans(void *inputBuf, size_t inputBufBytes, void *outputBuf, size_t outputBufBytes);

        static void getSmChanDeviceHandle(void *d_ptr, std::vector<mscclpp::SmChannel> &smChans);

        static std::shared_ptr<std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>>>
        getSmChanDeviceHandleAsync(void *d_ptr, std::vector<mscclpp::SmChannel> &smChans, cudaStream_t stream);

        int getWorldSize() { return p_.ep_world_size; }

        int getRank() { return p_.rank; }

        int getNumRanksPerNode() { return p_.num_ranks_per_node; }

        bool isCrossNode() { return getWorldSize() > getNumRanksPerNode(); }

        void allGather(void* allData, int size);

        void broadcast(std::vector<char>& data);

    private:
        void setupConnections();
        void setupProxyService();
        SemaphoreMap setupSemaphores();

    private:
        Params p_{};
        mscclpp::Transport ibTransport_;
        std::unique_ptr<mscclpp::Communicator> comm_;
        std::vector<std::shared_ptr<mscclpp::Connection>> connections_;
        std::shared_ptr<mscclpp::ProxyService> chanService_;
        std::unordered_map<size_t, std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>> smSemaphores_;
    };

    using Handle = mscclpp::DeviceHandle<mscclpp::SmChannel>;
    using HandleArrPtr = std::unique_ptr<Handle[], mscclpp::CudaDeleter<Handle[]>>;
    using ProxyHandle = mscclpp::DeviceHandle<mscclpp::ProxyChannel>;
    using ProxyHandleArrPtr = std::unique_ptr<ProxyHandle[], mscclpp::CudaDeleter<ProxyHandle[]>>;

    struct CommBuffBase
    {
        CommBuffBase() = default;

        CommBuffBase(int64_t _send_num_elems, int64_t _recv_num_elems, int64_t _rank, int64_t _num_ranks_per_node) :
            send_num_elems_{_send_num_elems},
            recv_num_elems_{_recv_num_elems},
            rank{_rank},
            num_ranks_per_node{_num_ranks_per_node}
            {}

        void set_recv_offset(int64_t offset) {
            recv_offset_in_bytes_ = offset;
        }

        void set_send_offset(int64_t offset) {
            send_offset_in_bytes_ = offset;
        }

        virtual void set_recv(char *_recv_buf) = 0;

        virtual void set_send(char* _send_buf) = 0;

        void set_smChans(Handle* chans) {
            smChans_ = chans;
        }

        void set_proxyChans(ProxyHandle* chans) {
            proxyChans_ = chans;
        }

    public:
        int64_t send_bytes{};
        int64_t recv_bytes{};

        int64_t recv_num_elems_{};
        int64_t recv_offset_in_bytes_{};

        int64_t send_num_elems_{};
        int64_t send_offset_in_bytes_{};

        Handle *smChans_{};
        ProxyHandle *proxyChans_{};

        int64_t rank{};
        int64_t num_ranks_per_node{};
    };

    template <typename T>
    struct CommBuffHost : public CommBuffBase
    {
        using Dtype = T;

        CommBuffHost() = default;

        CommBuffHost(int64_t _send_num_elems, int64_t _recv_num_elems, int64_t _rank, int64_t _num_ranks_per_node) : CommBuffBase(_send_num_elems, _recv_num_elems, _rank, _num_ranks_per_node) 
        {
            send_bytes = sizeof(T) * send_num_elems_;
            recv_bytes = sizeof(T) * recv_num_elems_;
        }

        void set_recv(char *_recv_buf) override {
            recv_buf = (T*)(_recv_buf + recv_offset_in_bytes_);
        }

        void set_send(char* _send_buf) override {
            send_buf = (T*)(_send_buf + send_offset_in_bytes_);
        }

    public:
        T *send_buf{};
        T *recv_buf{};
    };

    template <typename T>
    struct CommBuff
    {
        CommBuff(const CommBuffHost<T>& host) :
            recv_offset_in_bytes_{host.recv_offset_in_bytes_},
            send_offset_in_bytes_{host.send_offset_in_bytes_},
            smChans_{host.smChans_},
            proxyChans_{host.proxyChans_},
            send_buf{host.send_buf},
            recv_buf{host.recv_buf}
            {}


        __device__ void smChan_put(int remote_rank, uint64_t targetOffset, uint64_t originOffset, uint64_t originBytes, uint32_t threadId, uint32_t numThreads, int alignment) {
            if (alignment == 16)
            {
                smChans_[remote_rank].put<16>(recv_offset_in_bytes_ + targetOffset, send_offset_in_bytes_ + originOffset, originBytes, threadId, numThreads);
            }
            else if (alignment == 8)
            {
                smChans_[remote_rank].put<8>(recv_offset_in_bytes_ + targetOffset, send_offset_in_bytes_ + originOffset, originBytes, threadId, numThreads);
            }
            else if (alignment == 4)
            {
                smChans_[remote_rank].put<4>(recv_offset_in_bytes_ + targetOffset, send_offset_in_bytes_ + originOffset, originBytes, threadId, numThreads);
            } else if (alignment == 2) {
                smChans_[remote_rank].put<2>(recv_offset_in_bytes_ + targetOffset, send_offset_in_bytes_ + originOffset, originBytes, threadId, numThreads);
            } else {
                printf("Not supported alignment");
            }
        }
    
        __device__ void smChan_wait(int remote_rank) {
            smChans_[remote_rank].wait(-1);
        }
    
        __device__ void smChan_signal(int remote_rank) {
            smChans_[remote_rank].signal();
        }
    
        __device__ void proxyChan_putWithSignalAndFlush(int remote_rank, uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
            proxyChans_[remote_rank].putWithSignalAndFlush(recv_offset_in_bytes_ + dstOffset, send_offset_in_bytes_ + srcOffset, size);
        }
    
        __device__ void proxyChan_putWithSignal(int remote_rank, uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
            proxyChans_[remote_rank].putWithSignal(recv_offset_in_bytes_ + dstOffset, send_offset_in_bytes_ + srcOffset, size);
        }
    
        __device__ void proxyChan_wait(int remote_rank) {
            proxyChans_[remote_rank].wait(-1);
        }
    
        __device__ void proxyChan_signal(int remote_rank) {
            proxyChans_[remote_rank].signal();
        }
    
        __device__ void proxyChan_flush(int remote_rank) {
            proxyChans_[remote_rank].flush();
        }

    public:
        int64_t recv_offset_in_bytes_;
        int64_t send_offset_in_bytes_;

        Handle *smChans_;
        ProxyHandle *proxyChans_;

        T *send_buf;
        T *recv_buf;
    };

    template <typename DispatchType, typename CombineType, typename ExpertScaleType>
    struct CommBuffs
    {
        struct Params
        {
            int topk;
            int64_t max_num_tokens_per_gpu;
            int64_t hidden_size;
            int input_scales_dim{};
            int ep_world_size;
            // max token count per DP rank
            int max_num_stages;
            MscclppCommunicator *comm;

            std::string str() {
                std::ostringstream oss;
                oss << "Params(topk: " << topk << " max_num_tokens_per_gpu: " << max_num_tokens_per_gpu << " hidden_size: " << hidden_size << ")";
                return oss.str();
            }
        };

        CommBuffs(Params p);

        CommBuffHost<ExpertScaleType> expert_scales;
        CommBuffHost<int> expert_indices;
        CommBuffHost<DispatchType> input_hidden;
        CommBuffHost<float> input_scales;
        CommBuffHost<CombineType> output_hidden;
        CommBuffHost<int> dispatch_comm_plan;
        CommBuffHost<int> combine_comm_plan;

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
}
