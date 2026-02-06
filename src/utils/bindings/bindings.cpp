#include <torch/library.h>

#include "bindings/utils_ops.h"
#include "src/registration.h"

TORCH_LIBRARY(utils, m) {
  eps::register_rms_norm_op_ops(m);
  eps::register_topk_sigmoid_ops(m);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
