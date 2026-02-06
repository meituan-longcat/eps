#pragma once

#include <torch/library.h>

namespace eps {
void register_scheduler_ops(torch::Library &m);
} // namespace pplx
