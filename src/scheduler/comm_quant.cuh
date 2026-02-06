#pragma once

#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>

#include "src/scheduler/common.cuh"
#include "src/common/common.cuh"
#include "src/communication/msccl_comm.cuh"

namespace eps
{

    template <typename SrcType_, typename DstType_>
    class CommQuant {
    public:
        // Should satisfy the constraint specified in Context::getWorkspaceSize
        virtual size_t getWorkspaceSize() = 0;

        virtual const DstType_ *run(const SrcType_ *input, void *ws, cudaStream_t stream) = 0;

        virtual const float* get_scale() = 0;
    };

    template <typename SrcType_, typename DstType_>
    class NonCommQuant: public CommQuant<SrcType_, DstType_>
    {
    public:
        using SrcType = SrcType_;
        using DstType = DstType_;

        size_t getWorkspaceSize() override
        {
            return 0;
        }

        const DstType *run(const SrcType *input, void *ws, cudaStream_t stream) override;

        const float* get_scale() { return nullptr; }
    };

    template <typename SrcType_, typename DstType_>
    class StaticCommQuant: public CommQuant<SrcType_, DstType_>
    {
    public:
        using SrcType = SrcType_;
        using DstType = DstType_;

        struct Params
        {
            int num_tokens;
            int64_t hidden_size;
            const float *scale;
        };

        StaticCommQuant(Params p) : p_{p} {}

        size_t getWorkspaceSize() override
        {
            return ALIGN(sizeof(DstType) * p_.num_tokens * p_.hidden_size);
        }

        const DstType *run(const SrcType *input, void *ws, cudaStream_t stream) override;

        const float* get_scale() { return p_.scale; }

    private:
        Params p_{};
    };

    template <typename SrcType_, typename DstType_>
    class DynamicCommQuant: public CommQuant<SrcType_, DstType_>
    {
    public:
        using SrcType = SrcType_;
        using DstType = DstType_;

        struct Params
        {
            int num_tokens;
            int64_t hidden_size;
            CommBuff<float> comm_buff;
            float *scale;
            float *gathered_scales;
            int rank;
            int ep_world_size;
        };

        DynamicCommQuant(Params p) : p_{p} {}

        size_t getWorkspaceSize() override
        {
            return ALIGN(sizeof(DstType) * p_.num_tokens * p_.hidden_size);
        }

        const DstType *run(const SrcType *input, void *ws, cudaStream_t stream) override;

        const float* get_scale() { return p_.scale; }
    private:
        Params p_{};
    };

}
