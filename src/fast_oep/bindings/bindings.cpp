#include <torch/library.h>

#include "bindings/all_to_all_ops.h"
#include "src/registration.h"

TORCH_LIBRARY(fast_oep, m) {
  fast_oep::register_all_to_all_ops(m);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
