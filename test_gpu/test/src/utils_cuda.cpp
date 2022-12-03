#include "utils_cuda.h"
#include <array>

template <typename Scalar>
void PerfCuda::perfCopy(Scalar *dst, Scalar *src, size_t length, cudaMemcpyKind direction, const std::string &msg) {
    meas(msg);
    checkCuda(cudaMemcpy(dst, src, sizeof(*src) * length, direction));
    stop();
}

template <typename Scalar>
void PerfCuda::perfCopy2DTimes(Scalar* dst, size_t dst_pitch_bytes, Scalar* src, size_t src_pitch_bytes, size_t cols, size_t rows, const std::string& msg) {
    static constexpr size_t s_num_of_iterations = 5;

    s_indent_(std::cout, stack_);
    std::cout << msg << "..." << std::endl;
    ++stack_;

    std::array<float, s_num_of_iterations> milliseconds_array = {};

    size_t cols_bytes = cols * sizeof(Scalar);
    for (size_t i = 0; i < s_num_of_iterations; ++i)
    {
        auto & eventTuple = cudaEventTuple{};
        checkCuda(cudaEventCreate(&eventTuple.startEvent));
        checkCuda(cudaEventCreate(&eventTuple.stopEvent));
        checkCuda(cudaEventRecord(eventTuple.startEvent));

        // Copy from host to device
        checkCuda(cudaMemcpy2D(dst, dst_pitch_bytes, src, src_pitch_bytes, cols_bytes, rows, cudaMemcpyHostToDevice));

        // Copy from device to host
        checkCuda(cudaMemcpy2D(src, src_pitch_bytes, dst, dst_pitch_bytes, cols_bytes, rows, cudaMemcpyDeviceToHost));

        checkCuda(cudaEventRecord(eventTuple.stopEvent));
        checkCuda(cudaEventSynchronize(eventTuple.stopEvent));
        checkCuda(cudaEventElapsedTime(&milliseconds_array[i], eventTuple.startEvent, eventTuple.stopEvent));
    }

    float median_millisecond = milliseconds_array[s_num_of_iterations / 2];
    double time_median = ((double)median_millisecond) / ((double)1E3);
    double time_avg = 0.0;

    std::cout << "Done!\t[";
    for (size_t i = 0; i < s_num_of_iterations - 1; ++i)
    {
        float millisecond = milliseconds_array[i];
        double time = ((double)millisecond) / ((double)1E3);
        time_avg += time;
        std::cout << time << "s, ";
    }
    float millisecond = milliseconds_array[s_num_of_iterations - 1];
    double time_last = ((double)millisecond) / ((double)1E3);
    time_avg += time_last;
    std::cout << time_last << "s]";

    time_avg /= s_num_of_iterations;

    std::cout << " Median: " << time_median << "s, Avg: " << time_avg << "s" << std::endl;
    --stack_;
}

template void PerfCuda::perfCopy<float>(float *dst, float *src, size_t length, cudaMemcpyKind direction, const std::string &msg);
template void PerfCuda::perfCopy<double>(double *dst, double *src, size_t length, cudaMemcpyKind direction, const std::string &msg);

template void PerfCuda::perfCopy2DTimes<float>(float* dst, size_t dst_pitch_bytes, float* src, size_t src_pitch_bytes, size_t cols, size_t rows, const std::string& msg);
template void PerfCuda::perfCopy2DTimes<double>(double* dst, size_t dst_pitch_bytes, double* src, size_t src_pitch_bytes, size_t cols, size_t rows, const std::string& msg);