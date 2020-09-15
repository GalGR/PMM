#pragma once

#include <cuda_runtime_api.h>
#include <iostream>
#include <assert.h>
#include <stddef.h>
#include <queue>
#include <exception>

inline cudaError_t checkCuda(cudaError_t stat) {
    if (stat != cudaSuccess) {
        std::cerr << "CUDA runtime synchronous error: " << cudaGetErrorString(stat) << std::endl;
        #if defined(NDEBUG)
            exit(1);
        #else
            assert(stat == cudaSuccess);
        #endif
    }
    return stat;
}

inline cudaError_t checkAsyncCuda(cudaError_t stat) {
    if (stat != cudaSuccess) {
        std::cerr << "CUDA runtime asyncronous error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        #if defined(NDEBUG)
            exit(1);
        #else
            assert(stat == cudaSuccess);
        #endif
    }
    return stat;
}

inline void checkAsyncCuda() {
    #if !defined(NDEBUG)
        cudaError_t stat = cudaDeviceSynchronize();
        if (stat != cudaSuccess) {
            std::cerr << "CUDA runtime asyncronous error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        }
        assert(stat == cudaSuccess);
    #endif
}

class PerfCuda {
private:
    struct cudaEventTuple {
        cudaEvent_t startEvent, stopEvent;
    };
    std::queue<cudaEventTuple> eventQ;
    size_t stack_ = 0;

    inline static void s_indent_(std::ostream &out, size_t stack) {
        for (size_t i = 0; i < stack; ++i) {
            out << '\t';
        }
    }

public:
    void meas(const std::string &msg) {
        eventQ.push(cudaEventTuple{});
        auto & eventTuple = eventQ.back();
        checkCuda(cudaEventCreate(&eventTuple.startEvent));
        checkCuda(cudaEventCreate(&eventTuple.stopEvent));
        checkCuda(cudaEventRecord(eventTuple.startEvent));
        s_indent_(std::cout, stack_);
        std::cout << msg << "..." << std::endl;
        ++stack_;
    }

    void stop() {
        auto &eventTuple = eventQ.front();
        checkCuda(cudaEventRecord(eventTuple.stopEvent));
        checkCuda(cudaEventSynchronize(eventTuple.stopEvent));
        float milliseconds;
        checkCuda(cudaEventElapsedTime(&milliseconds, eventTuple.startEvent, eventTuple.stopEvent));
        double time = ((double)milliseconds) / ((double)1E3);
        std::cout << "Done!\t" << time << "s" << std::endl;
        eventQ.pop();
        --stack_;
    }

    void error(const std::string &msg) {
        auto &eventTuple = eventQ.front();
        checkCuda(cudaEventRecord(eventTuple.stopEvent));
        checkCuda(cudaEventSynchronize(eventTuple.stopEvent));
        float milliseconds;
        checkCuda(cudaEventElapsedTime(&milliseconds, eventTuple.startEvent, eventTuple.stopEvent));
        double time = ((double)milliseconds) / ((double)1E3);
        std::cerr << "Error\t" << time << "s" << ": " << msg << std::endl;
        eventQ.pop();
        --stack_;
        #if defined(NDEBUG)
            exit(1);
        #else
            assert(0);
        #endif
    }

    void error(const std::exception &e, const std::string &msg) {
        auto &eventTuple = eventQ.front();
        checkCuda(cudaEventRecord(eventTuple.stopEvent));
        checkCuda(cudaEventSynchronize(eventTuple.stopEvent));
        float milliseconds;
        checkCuda(cudaEventElapsedTime(&milliseconds, eventTuple.startEvent, eventTuple.stopEvent));
        double time = ((double)milliseconds) / ((double)1E3);
        std::cerr << "Error\t" << time << "s" << ": " << msg << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        eventQ.pop();
        --stack_;
        #if defined(NDEBUG)
            exit(1);
        #else
            assert(0);
        #endif
    }

    void meas_async(const std::string &msg) {
        #if !defined(NDEBUG)
            eventQ.push(cudaEventTuple{});
            auto & eventTuple = eventQ.back();
            checkCuda(cudaEventCreate(&eventTuple.startEvent));
            checkCuda(cudaEventCreate(&eventTuple.stopEvent));
            checkCuda(cudaEventRecord(eventTuple.startEvent));
            s_indent_(std::cout, stack_);
            std::cout << msg << "..." << std::endl;
            ++stack_;
        #endif
    }

    void stop_async() {
        #if !defined(NDEBUG)
            auto &eventTuple = eventQ.front();
            checkCuda(cudaEventRecord(eventTuple.stopEvent));
            checkCuda(cudaEventSynchronize(eventTuple.stopEvent));
            float milliseconds;
            checkCuda(cudaEventElapsedTime(&milliseconds, eventTuple.startEvent, eventTuple.stopEvent));
            double time = ((double)milliseconds) / ((double)1E3);
            std::cout << "Done!\t" << time << "s" << std::endl;
            eventQ.pop();
            --stack_;
        #endif
    }

    void error_async(const std::string &msg) {
        #if !defined(NDEBUG)
            auto &eventTuple = eventQ.front();
            checkCuda(cudaEventRecord(eventTuple.stopEvent));
            checkCuda(cudaEventSynchronize(eventTuple.stopEvent));
            float milliseconds;
            checkCuda(cudaEventElapsedTime(&milliseconds, eventTuple.startEvent, eventTuple.stopEvent));
            double time = ((double)milliseconds) / ((double)1E3);
            std::cerr << "Error\t" << time << "s" << ": " << msg << std::endl;
            eventQ.pop();
            --stack_;
            assert(0);
        #endif
    }

    void error_async(const std::exception &e, const std::string &msg) {
        #if !defined(NDEBUG)
            auto &eventTuple = eventQ.front();
            checkCuda(cudaEventRecord(eventTuple.stopEvent));
            checkCuda(cudaEventSynchronize(eventTuple.stopEvent));
            float milliseconds;
            checkCuda(cudaEventElapsedTime(&milliseconds, eventTuple.startEvent, eventTuple.stopEvent));
            double time = ((double)milliseconds) / ((double)1E3);
            std::cerr << "Error\t" << time << "s" << ": " << msg << std::endl;
            std::cerr << "Exception: " << e.what() << std::endl;
            eventQ.pop();
            --stack_;
            assert(0);
        #endif
    }

    template <typename Scalar>
    void perfCopy(Scalar *dst, Scalar *src, size_t length, cudaMemcpyKind direction, const std::string &msg);
};