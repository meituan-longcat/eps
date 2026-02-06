#pragma once

#include <torch/library.h>

namespace eps {
void register_communication_ops(torch::Library &m);
} // namespace eps
