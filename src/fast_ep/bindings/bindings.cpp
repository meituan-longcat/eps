#include <torch/library.h>

#include "src/registration.h"
#include "bindings/all_to_all_ops.h"

TORCH_LIBRARY(fast_ep, m) {
  fast_ep::register_all_to_all_ops(m);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
