#pragma once

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

static inline size_t ALIGN(const size_t input) {
  static constexpr size_t ALIGNMENT = 256;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}
