#pragma once

namespace eps
{

    template <typename T, typename ExpertScalesType, typename ExpertIndicesType>
    struct TopkSigmoidParams
    {
        const T *scores;
        const float *bias;
        ExpertScalesType *expert_scales;
        ExpertIndicesType *expert_indices;
        int num_tokens;
        int num_experts;
        int num_expert_groups;
        int topk_groups;
        int topk;
        float route_scale;
    };

    template <typename T, typename ExpertScalesType, typename ExpertIndicesType>
    void topk_sigmoid(TopkSigmoidParams<T, ExpertScalesType, ExpertIndicesType> p, cudaStream_t stream);

}