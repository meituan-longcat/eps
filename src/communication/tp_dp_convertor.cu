#include "src/communication/tp_dp_convertor.cuh"

#include "src/communication/reducescatter_runner.cuh"
#include "src/communication/allgather_runner.cuh"
#include "src/common/debug.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace eps
{

    template <typename T>
    TPDPConvertorBuff<T>::TPDPConvertorBuff(TPDPConvertorParams p) : host_buff_(p.getMaxNumElems(), p.getMaxNumElems(), -1, -1) {
        host_buff_.set_send_offset(0);
        host_buff_.set_recv_offset(0);
        int64_t send_bytes = host_buff_.send_bytes;

        if (send_bytes) {
            cudaMalloc(&_raw_send_buf_, send_bytes);
            cudaMemset(_raw_send_buf_, -23, send_bytes);
            send_buf_ = TArrPtr{_raw_send_buf_};
        }
        const int world_size = p.comm->getWorldSize();

        cudaMalloc(&smChans, (sizeof(Handle) * p.attn_tp_size) * NUM_CHANNELS_PER_CONNECTION);
        smChansHandle_ = HandleArrPtr{smChans};
        
        std::vector<mscclpp::SmChannel> smChans_full_world;
        std::tie(smChans_full_world, proxyChans_, smSemaphores_) = p.comm->createChans(_raw_send_buf_, send_bytes, nullptr, 0);

        for (size_t i = 0; i < NUM_CHANNELS_PER_CONNECTION; ++i) {
            const int base_tp_rank = p.attn_tp_size * (p.global_rank / p.attn_tp_size);
            int base = i * world_size + base_tp_rank;
            for (int j = 0; j < p.attn_tp_size; ++j) {
                smChans_.emplace_back(smChans_full_world[base + j]);
            }
        }
        MscclppCommunicator::getSmChanDeviceHandle(smChans, smChans_);

        host_buff_.set_send(_raw_send_buf_);
        host_buff_.set_recv(_raw_send_buf_);
        host_buff_.set_smChans(smChans);
    }
    

    template <typename T>
    TPDPConvertor<T>::TPDPConvertor(TPDPConvertorParams p) : p_{p}
    {
        tpdp_convertor_buff_ = std::make_unique<TPDPConvertorBuff<T>>(p);
    }

    template <typename T>
    std::tuple<int, int> TPDPConvertor<T>::get_dp_num_tokens_and_offset(int tp_num_tokens)
    {
        // 三目运算符优先级
        // tp_num_tokens 是每个 dp_rank 之内共享的 token 数
        const int tp_rank = p_.global_rank % p_.attn_tp_size;
        int dp_num_tokens = tp_num_tokens / p_.attn_tp_size + ((tp_rank < tp_num_tokens % p_.attn_tp_size) ? 1 : 0);
        int offset = (tp_num_tokens / p_.attn_tp_size) * tp_rank + std::min(tp_rank, tp_num_tokens % p_.attn_tp_size);
        return {dp_num_tokens, offset};
    }

    template <typename T>
    void TPDPConvertor<T>::setBuf(T **buf) {
        *buf = (T*)tpdp_convertor_buff_->getBuf();
    }

    template <typename T>
    TPDPConvertor<T>::ReduceScatterContext TPDPConvertor<T>::get_reduce_scatter_context(int tp_num_tokens, int num_blocks)
    {
        
        auto [dp_num_tokens, offset] = get_dp_num_tokens_and_offset(tp_num_tokens);
        return ReduceScatterContext{
            .input = (T*)tpdp_convertor_buff_->getBuf(),
            .input_rows = tp_num_tokens,
            .output = (T*)tpdp_convertor_buff_->getBuf() + offset * p_.hidden_size,
            .output_rows = dp_num_tokens,
            .output_row_offset = offset,
            .hidden_size = p_.hidden_size,
            .rank = p_.global_rank,
            .num_ranks_per_node = p_.comm->getNumRanksPerNode(),
            .num_blocks = num_blocks
        };
    }
    template <typename T>
    void TPDPConvertor<T>::reduce_scatter(ReduceScatterContext context, cudaStream_t stream)
    {   
        typename ReduceScatterRunner::Params p {
            .global_rank = p_.global_rank, 
            .attn_tp_size = p_.attn_tp_size,
            .num_blocks = context.num_blocks
        };
        ReduceScatterRunner runner(p);
        runner.run<T>(tpdp_convertor_buff_->smChans, context.input_rows, p_.hidden_size, stream);
    }

    template <typename T>
    TPDPConvertor<T>::AllGatherContext TPDPConvertor<T>::get_all_gather_context(int tp_num_tokens, int64_t hidden_size, int num_blocks)
    {
        if (hidden_size > p_.hidden_size) {
            throw std::runtime_error(std::to_string(hidden_size) + " should be smaller than or equal to" + std::to_string(p_.hidden_size));
        }

        auto [dp_num_tokens, offset] = get_dp_num_tokens_and_offset(tp_num_tokens);
        return AllGatherContext{
            .input = (T*)tpdp_convertor_buff_->getBuf() + offset * hidden_size,
            .input_rows = dp_num_tokens,
            .input_row_offset = offset,
            .output = (T*)tpdp_convertor_buff_->getBuf(),
            .output_rows = tp_num_tokens,
            .hidden_size = hidden_size,
            .rank = p_.global_rank,
            .num_ranks_per_node = p_.comm->getNumRanksPerNode(),
            .num_blocks = num_blocks
        };
       
    }
    template <typename T>
    void TPDPConvertor<T>::all_gather(AllGatherContext context, cudaStream_t stream)
    {   
        typename AllGatherRunner::Params p{
            .global_rank = p_.global_rank,
            .attn_tp_size = p_.attn_tp_size,
            .num_blocks = context.num_blocks
        };
        AllGatherRunner runner(p);
        runner.run<T>(tpdp_convertor_buff_->smChans, context.output_rows, context.hidden_size, stream);
    }

    template class TPDPConvertor<float>;
    template class TPDPConvertor<half>;
#ifdef ENABLE_BF16
    template class TPDPConvertor<__nv_bfloat16>;
#endif
}