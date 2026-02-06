#include "src/utils/topk_sigmoid.cuh"

#include <stdexcept>

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

#include "src/common/cuda_utils.cuh"

namespace cg = cooperative_groups;

namespace eps
{
    template <typename T>
    __device__ __forceinline__ T sigmoid(T x)
    {
        float r = 1.0f / (1.0f + expf(-(float)x));
        return r;
    }

    struct Pack2
    {
        int index;
        float score;
    };

    struct Pack3
    {
        int expert_index;
        float score;
        float original_score;
    };

    template <typename PackType, typename TileType>
    __device__ __forceinline__ PackType max(PackType pack, TileType tile)
    {
        for (int offset = tile.size() / 2; offset > 0; offset /= 2)
        {
            PackType other;
            other = tile.shfl_down(pack, offset);
            if (other.score > pack.score)
            {
                pack = other;
            }
        }

        pack = tile.shfl(pack, 0);
        return pack;
    }

    template<typename T>
    struct minimum;

    template<>
    struct minimum<float> {
        // static constexpr float val = FLT_MIN; // not correct, Minimum normalized `positive` floating-point number
        static constexpr float val = -FLT_MAX;
    };

    template <typename T, typename TileType>
    inline __device__ T get_group_score(T *scores, int eles_per_thread, TileType group)
    {
        Pack2 pack2s[2];
        #pragma unroll
        for (int pack_idx = 0; pack_idx < 2; pack_idx++)
        {
            T thread_max = minimum<T>::val;
            int expert_index_in_group = -1;
            for (int i = 0; i < eles_per_thread; i++)
            {
                bool selected = false;
                for (int j = 0; j < pack_idx; j++)
                {
                    if (pack2s[j].index == i + eles_per_thread * group.thread_rank())
                    {
                        selected = true;
                        break;
                    }
                }

                if (!selected && scores[i] > thread_max)
                {
                    thread_max = scores[i];
                    expert_index_in_group = i + eles_per_thread * group.thread_rank();
                }
            }

            Pack2 pack{.index = expert_index_in_group, .score = thread_max};

            pack2s[pack_idx] = max(pack, group);
        }

        T group_score{};
        #pragma unroll
        for (int pack_idx = 0; pack_idx < 2; pack_idx++)
        {
            group_score += pack2s[pack_idx].score;
        }
        return group_score;
    }

    template <typename T>
    inline __device__ bool select_group(cg::coalesced_group g, int expert_group_index, T group_score, int topk_groups)
    {
        bool group_selected = false;
        for (int i = 0; i < topk_groups; i++)
        {
            Pack2 pack{.index = expert_group_index, .score = group_selected ? minimum<T>::val : group_score};

            pack = max(pack, g);

            group_selected = group_selected || (pack.index == expert_group_index);
        }
        return group_selected;
    }

    template <typename T>
    inline __device__ void select_expert(T *scores,
                                         const T *original_scores,
                                         int eles_per_thread,
                                         int *expert_indices,
                                         T *selected_original_scores,
                                         int topk,
                                         int expert_begin,
                                         cg::coalesced_group g)
    {
        for (int topk_idx = 0; topk_idx < topk; topk_idx++)
        {
            T thread_max = minimum<T>::val;
            T original_score;
            int expert_index = -1;
            for (int i = 0; i < eles_per_thread; i++)
            {
                if (scores[i] > thread_max)
                {
                    thread_max = scores[i];
                    original_score = original_scores[i];
                    expert_index = i + expert_begin;
                }
            }

            Pack3 pack{.expert_index = expert_index, .score = thread_max, .original_score = original_score};

            pack = max(pack, g);

            expert_indices[topk_idx] = pack.expert_index;
            selected_original_scores[topk_idx] = pack.original_score;
            if (expert_begin <= pack.expert_index && pack.expert_index < expert_begin + eles_per_thread)
            {
                scores[pack.expert_index - expert_begin] = minimum<T>::val;
            }
        }
    }

    template <typename T, typename ExpertScalesType, typename ExpertIndicesType, int num_experts, int n_groups, int topk>
    __global__ void topk_sigmoid_kernel(TopkSigmoidParams<T, ExpertScalesType, ExpertIndicesType> p)
    {
        static_assert(num_experts % n_groups == 0);

        auto grid = cg::this_grid();
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        auto group = cg::tiled_partition<32 / n_groups>(warp);

        /* warp.meta_group_size() == 16, group.meta_group_size() == 8*/
        int block_idx = grid.block_index().x;
        int num_blocks_in_grid = grid.dim_blocks().x;
        int warp_idx = warp.meta_group_rank();
        int num_warps_in_block = warp.meta_group_size();

        constexpr int eles_per_thread = num_experts / 32;
        static_assert(eles_per_thread % VectorizedType<T>::ItemsPerVec == 0);

        float scores[eles_per_thread];
        float original_scores[eles_per_thread];
        float bias[eles_per_thread];

        int token_idx = block_idx * num_warps_in_block + warp_idx, 
                offset = num_blocks_in_grid * num_warps_in_block;
        for (; token_idx < p.num_tokens; token_idx += offset)
        {
            const T *scores_ptr = p.scores + token_idx * num_experts + warp.thread_rank() * eles_per_thread;
            const VectorizedType<T> *scores_ptr_v = reinterpret_cast<const VectorizedType<T> *>(scores_ptr);

            int item_idx = 0;
            #pragma unroll
            for (int v_idx = 0; v_idx < eles_per_thread / VectorizedType<T>::ItemsPerVec; v_idx++)
            {
                auto v = *(scores_ptr_v + v_idx);
                for (int i = 0; i < VectorizedType<T>::ItemsPerVec; i++)
                {
                    scores[item_idx] = v(i);
                    item_idx++;
                }
            }

            // TODO: shared memory
            const float *bias_ptr = p.bias + warp.thread_rank() * eles_per_thread;
            const VectorizedType<float> *bias_ptr_v = reinterpret_cast<const VectorizedType<float> *>(bias_ptr);

            item_idx = 0;
            #pragma unroll
            for (int v_idx = 0; v_idx < eles_per_thread / VectorizedType<float>::ItemsPerVec; v_idx++)
            {
                auto v = *(bias_ptr_v + v_idx);
                for (int i = 0; i < VectorizedType<float>::ItemsPerVec; i++)
                {
                    bias[item_idx] = v(i);
                    item_idx++;
                }
            }

            #pragma unroll
            for (int i = 0; i < eles_per_thread; i++)
            {
                scores[i] = sigmoid(scores[i]);
                original_scores[i] = scores[i];
                scores[i] += bias[i];
            }

            float group_score = get_group_score(scores, eles_per_thread, group);

            bool group_selected = false;
            if (group.thread_rank() == 0)
            {
                int expert_group_index = group.meta_group_rank();
                cg::coalesced_group g = cg::coalesced_threads();
                group_selected = select_group(g, expert_group_index, group_score, p.topk_groups);
            }
            group_selected = group.shfl(group_selected, 0);

            if (group_selected)
            {
                int expert_indices[topk];
                float selected_original_scores[topk];
                cg::coalesced_group g = cg::coalesced_threads();
                int eles_per_group = num_experts / n_groups;
                int expert_begin = eles_per_thread * group.thread_rank() + group.meta_group_rank() * eles_per_group;

                select_expert(scores, original_scores, eles_per_thread, expert_indices, selected_original_scores, topk, expert_begin, g);

                if (g.thread_rank() == 0)
                {
                    float sum{};
                    #pragma unroll
                    for (int topk_idx = 0; topk_idx < topk; topk_idx++)
                    {
                        sum += selected_original_scores[topk_idx];
                    }
                    float rescale = p.route_scale / sum;

                    #pragma unroll
                    for (int topk_idx = 0; topk_idx < topk; topk_idx++)
                    {
                        selected_original_scores[topk_idx] *= rescale;
                    }

                    ExpertScalesType *expert_scales_ptr = p.expert_scales + token_idx * topk;
                    ExpertIndicesType *expert_indices_ptr = p.expert_indices + token_idx * topk;
                    #pragma unroll
                    for (int topk_idx = 0; topk_idx < topk; topk_idx++)
                    {
                        expert_scales_ptr[topk_idx] = (ExpertScalesType)selected_original_scores[topk_idx];
                        expert_indices_ptr[topk_idx] = expert_indices[topk_idx];
                    }
                }
            }
        }
    }

    template <typename T, typename ExpertScalesType, typename ExpertIndicesType>
    void topk_sigmoid(TopkSigmoidParams<T, ExpertScalesType, ExpertIndicesType> p, cudaStream_t stream)
    {
        int block_size = 512;
        int num_warps = (block_size / 32);
        int num_blocks = (p.num_tokens + num_warps - 1) / num_warps;
        if (p.num_experts == 256 && p.num_expert_groups == 8 && p.topk == 8)
        {
            topk_sigmoid_kernel<T, ExpertScalesType, ExpertIndicesType, 256, 8, 8><<<num_blocks, block_size, 0, stream>>>(p);
        }
        else
        {
            throw std::runtime_error("Not supported num_experts: " + std::to_string(p.num_experts) + " num_expert_groups: " + std::to_string(p.num_expert_groups) + " topk: " + std::to_string(p.topk));
        }
    }
    template void topk_sigmoid(TopkSigmoidParams<float, float, int> p, cudaStream_t stream);

    template void topk_sigmoid(TopkSigmoidParams<half, half, int> p, cudaStream_t stream);

    template void topk_sigmoid(TopkSigmoidParams<__nv_bfloat16, __nv_bfloat16, int> p, cudaStream_t stream);

    template void topk_sigmoid(TopkSigmoidParams<__nv_bfloat16, float, int> p, cudaStream_t stream);

    template void topk_sigmoid(TopkSigmoidParams<float, float, int64_t> p, cudaStream_t stream);

    template void topk_sigmoid(TopkSigmoidParams<half, float, int64_t> p, cudaStream_t stream);

    template void topk_sigmoid(TopkSigmoidParams<__nv_bfloat16, float, int64_t> p, cudaStream_t stream);
}