#pragma once

#include <torch/library.h>

namespace eps {
void register_rms_norm_op_ops(torch::Library &m);
void register_topk_sigmoid_ops(torch::Library &m);
} // namespace utils
