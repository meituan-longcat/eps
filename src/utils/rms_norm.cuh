#pragma once

namespace eps {

template <typename T>
void rms_norm(T* out, const T* input, const T* scale, float eps, int m, int n, cudaStream_t stream);

}
