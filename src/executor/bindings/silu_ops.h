#pragma once

#include <torch/library.h>

namespace executor {
void register_silu_ops(torch::Library &m);
} // namespace executor
