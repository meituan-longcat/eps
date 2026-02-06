#pragma once

// Redirect upstream CUTLASS include to local extension header to avoid
// duplicate symbol definitions when both paths are included.
#include "src/gemm/cutlass_extensions/epilogue/collective/detail.hpp"
