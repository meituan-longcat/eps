#include <torch/library.h>

#include "bindings/communication_ops.h"
#include "src/registration.h"

TORCH_LIBRARY(communication, m) {
  eps::register_communication_ops(m);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
