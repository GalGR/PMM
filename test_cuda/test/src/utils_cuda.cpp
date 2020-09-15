#include "utils_cuda.h"

template <typename Scalar>
void PerfCuda::perfCopy(Scalar *dst, Scalar *src, size_t length, cudaMemcpyKind direction, const std::string &msg) {
    assert(sizeof(dst) == sizeof(src));
    meas(msg);
    checkCuda(cudaMemcpy(dst, src, sizeof(src) * length, direction));
    stop();
}

template void PerfCuda::perfCopy<float>(float *dst, float *src, size_t length, cudaMemcpyKind direction, const std::string &msg);
template void PerfCuda::perfCopy<double>(double *dst, double *src, size_t length, cudaMemcpyKind direction, const std::string &msg);