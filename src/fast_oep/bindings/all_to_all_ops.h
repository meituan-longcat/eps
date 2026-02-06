#pragma once

#include <torch/library.h>

namespace fast_oep {
void register_all_to_all_ops(torch::Library &m);
} // namespace pplx
