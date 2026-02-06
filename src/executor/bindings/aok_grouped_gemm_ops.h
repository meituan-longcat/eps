#pragma once

#include <torch/library.h>

namespace executor {
void register_aok_gemm_ops(torch::Library &m);
} // namespace executor
