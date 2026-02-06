#pragma once

#include <set>
#include <functional>

#include "src/communication/msccl_comm.cuh"

namespace eps
{

    struct TPDPConvertorParams
    {
        int global_rank;
        int tp_max_num_tokens;
        int attn_tp_size;
        int hidden_size;
        MscclppCommunicator *comm;
        size_t getMaxNumElems() const {
            return (size_t)attn_tp_size * tp_max_num_tokens * hidden_size;
        }
    };

    template <typename T>
    struct TPDPConvertorBuff
    {
        TPDPConvertorBuff(TPDPConvertorParams p);

        CommBuffHost<T> host_buff_;
        
        Handle *smChans;
        ProxyHandle *proxyChans;
        MscclppCommunicator::SemaphoreMap smSemaphores_;

        char* getBuf() const {return _raw_send_buf_;}

        using TArrPtr = std::unique_ptr<char[], mscclpp::CudaDeleter<char[]>>;
        char* _raw_send_buf_;
        TArrPtr send_buf_;
        TArrPtr recv_buf_;
        std::vector<mscclpp::SmChannel> smChans_;
        HandleArrPtr smChansHandle_;
        std::vector<mscclpp::DeviceHandle<mscclpp::ProxyChannel>> proxyChans_;
        ProxyHandleArrPtr proxyChansHandle_;

    };

    template <typename T>
    class TPDPConvertor
    {
    public:
        TPDPConvertor(TPDPConvertorParams p);

        void setBuf(T **buf);

        struct ReduceScatterContext
        {
            T *input;
            int input_rows;
            T *output;
            int output_rows;
            int output_row_offset;
            int64_t hidden_size;
            int rank;
            int num_ranks_per_node;
            int num_blocks;
        };

        ReduceScatterContext get_reduce_scatter_context(int tp_num_tokens, int num_blocks=45);

        void reduce_scatter(ReduceScatterContext context, cudaStream_t stream);

        struct AllGatherContext
        {
            T *input;
            int input_rows;
            int input_row_offset;
            T *output;
            int output_rows;
            int64_t hidden_size;
            int rank;
            int num_ranks_per_node;
            int num_blocks;
        };

        AllGatherContext get_all_gather_context(int tp_num_tokens, int64_t hidden_size, int num_blocks=28);

        void all_gather(AllGatherContext context, cudaStream_t stream);

    private:
        std::tuple<int, int> get_dp_num_tokens_and_offset(int tp_num_tokens);

    private:
        TPDPConvertorParams p_;
        std::unique_ptr<TPDPConvertorBuff<T>> tpdp_convertor_buff_;
    };
}
