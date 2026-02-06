#include <torch/library.h>

#include "bindings/scheduler_ops.h"
#include "src/registration.h"

TORCH_LIBRARY(eps, m) {
  eps::register_scheduler_ops(m);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
