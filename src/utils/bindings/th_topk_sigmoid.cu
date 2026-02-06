#include <torch/csrc/cuda/Stream.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/common/torch_utils.h"
#include "src/utils/topk_sigmoid.cuh"

using torch::Tensor;
using namespace eps;

namespace {

    void topk_sigmoid_op(Tensor scores,
                         Tensor bias,
                         Tensor expert_scales,
                         Tensor expert_indices,
                         int64_t num_tokens,
                         int64_t num_experts,
                         int64_t num_expert_groups,
                         int64_t topk_groups,
                         int64_t topk,
                         double route_scale)
    {
        EPS_CHECK_INPUT(scores, torch::kBFloat16);
        EPS_CHECK_INPUT(bias, torch::kFloat);
        EPS_CHECK_INPUT(expert_scales, torch::kFloat);
        if (expert_indices.scalar_type() == torch::kInt32) {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            topk_sigmoid(
                TopkSigmoidParams<__nv_bfloat16, float, int>{
                    .scores = get_ptr<__nv_bfloat16>(scores),
                    .bias = get_ptr<float>(bias),
                    .expert_scales = get_ptr<float>(expert_scales),
                    .expert_indices = get_ptr<int>(expert_indices),
                    .num_tokens = (int)num_tokens,
                    .num_experts = (int)num_experts,
                    .num_expert_groups = (int)num_expert_groups,
                    .topk_groups = (int)topk_groups,
                    .topk = (int)topk,
                    .route_scale = (float)route_scale},
                stream);
        } else if (expert_indices.scalar_type() == torch::kInt64) {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            topk_sigmoid(
                TopkSigmoidParams<__nv_bfloat16, float, int64_t>{
                    .scores = get_ptr<__nv_bfloat16>(scores),
                    .bias = get_ptr<float>(bias),
                    .expert_scales = get_ptr<float>(expert_scales),
                    .expert_indices = get_ptr<int64_t>(expert_indices),
                    .num_tokens = (int)num_tokens,
                    .num_experts = (int)num_experts,
                    .num_expert_groups = (int)num_expert_groups,
                    .topk_groups = (int)topk_groups,
                    .topk = (int)topk,
                    .route_scale = (float)route_scale},
                stream);
        } else {
            throw std::runtime_error("Not supported expert_indices data type.");
        }
    }

} 

namespace eps {

void register_topk_sigmoid_ops(torch::Library &m) {
    m.def("topk_sigmoid", &topk_sigmoid_op);
}

}
