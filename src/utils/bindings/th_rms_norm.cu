#include <torch/csrc/cuda/Stream.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/common/torch_utils.h"
#include "src/utils/rms_norm.cuh"

using torch::Tensor;
using namespace eps;

namespace {

    void rms_norm_op(Tensor output,
                  Tensor input,
                  Tensor scale,
                  double eps)
    {
        EPS_CHECK_INPUT(output, torch::kBFloat16);
        EPS_CHECK_INPUT(input, torch::kBFloat16);
        EPS_CHECK_INPUT(scale, torch::kBFloat16);
        if (output.dim() != 2 || input.dim() != 2) {
            throw std::runtime_error("output and input dim must be 2 but got " + std::to_string(output.dim()) + " and " + std::to_string(input.dim()));
        }

        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        rms_norm(
            get_ptr<__nv_bfloat16>(output),
            get_ptr<__nv_bfloat16>(input),
            get_ptr<__nv_bfloat16>(scale),
            eps,
            input.size(0),
            input.size(1),
            stream);
    }
}

namespace eps {

void register_rms_norm_op_ops(torch::Library &m) {
    m.def("rms_norm_op", &rms_norm_op);
}

}
