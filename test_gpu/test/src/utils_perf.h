#pragma once

#include <stddef.h>
#include "plf_nanotimer/plf_nanotimer.h"
#include <string>
#include <iostream>
#include <exception>

#define UTILS_PERF_MAKE_METHODS_(UNITS) \
private: \
    inline static void s_stop_##UNITS##_(plf::nanotimer &timer, size_t &stack) { \
        --stack; \
        s_indent_(std::cout, stack); \
        std::cout << "Done!\t" << timer.get_elapsed_##UNITS() << #UNITS << std::endl; \
    } \
    template <typename String_T> \
    inline static void s_stop_##UNITS##_(plf::nanotimer &timer, size_t &stack, const String_T &msg) { \
        --stack; \
        s_indent_(std::cout, stack); \
        std::cout << "Done!\t" << timer.get_elapsed_##UNITS() << #UNITS << ": " << msg << std::endl; \
    } \
    template <typename String_T> \
    inline static void s_error_##UNITS##_(plf::nanotimer &timer, size_t &stack, const String_T &msg, const std::exception &e) { \
        --stack; \
        s_indent_(std::cerr, stack); \
        std::cerr << "Error\t" << timer.get_elapsed_##UNITS() << #UNITS << ": " << msg << std::endl; \
        std::cerr << "Exception: " << e.what() << std::endl; \
    } \
    template <typename String_T> \
    inline static void s_error_##UNITS##_(plf::nanotimer &timer, size_t &stack, const String_T &msg) { \
        --stack; \
        s_indent_(std::cerr, stack); \
        std::cerr << "Error\t" << timer.get_elapsed_##UNITS() << #UNITS << ": " << msg << std::endl; \
    } \
public: \
    inline void stop_##UNITS() { \
        s_stop_##UNITS##_(timer_, stack_); \
    } \
    template <typename String_T> \
    inline void stop_##UNITS(const String_T &msg) { \
        s_stop_##UNITS##_(timer_, stack_, msg); \
    } \
    template <typename String_T> \
    inline void error_##UNITS(const String_T &msg) { \
        s_error_##UNITS##_(timer_, stack_, msg); \
    } \
    template <typename String_T> \
    inline void error_##UNITS(const std::exception &e, const String_T &msg) { \
        s_error_##UNITS##_(timer_, stack_, msg, e); \
    }

class PerfUtil {
private:
    plf::nanotimer timer_;
    size_t stack_ = 0;

    inline static void s_indent_(std::ostream &out, size_t stack) {
        for (size_t i = 0; i < stack; ++i) {
            out << '\t';
        }
    }
    template <typename String_T>
    inline static void s_meas_(plf::nanotimer &timer, size_t &stack, const String_T &msg) {
        timer.start();
        s_indent_(std::cout, stack);
        std::cout << msg << "..." << std::endl;
        ++stack;
    }
public:

    UTILS_PERF_MAKE_METHODS_(s)
    UTILS_PERF_MAKE_METHODS_(ms)
    UTILS_PERF_MAKE_METHODS_(us)
    UTILS_PERF_MAKE_METHODS_(ns)

public:
    template <typename String_T>
    inline void meas(const String_T &msg) {
        s_meas_(timer_, stack_, msg);
    }
    inline void stop() {
        stop_s();
    }
    template <typename String_T>
    inline void stop(const String_T &msg) {
        stop_s(msg);
    }
    template <typename String_T>
    inline void error(const String_T &msg) {
        error_s(msg);
    }
    template <typename String_T>
    inline void error(const std::exception &e, const String_T &msg) {
        error_s(e, msg);
    }
};