#include "pmm.cuh"

#include <Eigen/Dense>
#include <cublas_v2.h>
#include <array>
#include <limits>
#include <cmath>
#include <cstring>
#include <assert.h>
#include "utils_cuda.h"
#include "scalar_types.h"

enum PMMCudaStreams {
    PMM_UPWARDS_STREAM = 1,
    PMM_DOWNWARDS_STREAM,
    PMM_RIGHTWARDS_STREAM,
    PMM_LEFTWARDS_STREAM,
};

#include "pmm_geodesics_solve_kernel.inl"

#include "pmm_geodesics_solve_upwards.inl"
#include "pmm_geodesics_solve_downwards.inl"
#include "pmm_geodesics_solve_rightwards.inl"
#include "pmm_geodesics_solve_leftwards.inl"

#define BLOCK_DIM 32
#include "pmm_geodesics_solve_transpose.inl"

template <typename Scalar>
PMM_INLINE void pmm_geodesics_solve(
    size_t rows, size_t cols,
    int maxGridWidth,
    int maxThreads,
    int warpSize,
    size_t maxSharedMem,
    cublasHandle_t cublasHandle,
    std::array<Scalar*, 4> &d_C,
    const std::array<size_t, 4> &d_C_pitch_bytes,
    const std::array<size_t, 4> &d_C_pitch,
    const cudaTextureObject_t V,
    const std::vector<unsigned> &S,
    Scalar *p_D,
    std::array<Scalar*, 2> &d_D,
    const std::array<size_t, 2> &d_D_pitch_bytes,
    const std::array<size_t, 2> &d_D_pitch,
    size_t N,
    size_t numWarps,
    size_t omega
) {

    // // Create a D_row (row major) and D_col (column major) distance maps
    // Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > D_row(D.data(), data.rows, data.cols);
    // Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> D_col = D_row;

    // auto solve = [&] (
    //     const auto &coeff,
    //     const auto &V,
    //     auto &D,
    //     const std::array<size_t, 3> &is,
    //     const std::array<size_t, 3> &js,
    //     size_t i_c, size_t j_c, size_t len
    // ) {

    //     auto solve_quadratic = [&] (Scalar a, Scalar b, Scalar c) -> Scalar {
    //         Scalar a_2 = 2 * a;
    //         Scalar sqroot = std::sqrt(b * b - 4 * a * c);
    //         Scalar rhs = sqroot / a_2;
    //         Scalar lhs = -b / a_2;
    //         Scalar res = std::max(lhs - rhs, lhs + rhs);
    //         // assert(!(std::isnan(res) || res == 1.0 / 0.0));
    //         return res;
    //     };

    //     auto solve_dijkstra = [&] (const Eigen::Matrix<Scalar, 3, 1> &x1, const Eigen::Matrix<Scalar, 3, 1> &x2, Scalar d1, Scalar d2) -> Scalar {
    //         return std::min(d1 + x1.norm(), d2 + x2.norm());
    //     };

    //     Scalar d1 = D(is[1], js[1]);
    //     Scalar d2 = D(is[2], js[2]);

    //     Eigen::Matrix<Scalar, 3, 1> x0;
    //     x0 << V.X(is[0], js[0]), V.Y(is[0], js[0]), V.Z(is[0], js[0]);
    //     Eigen::Matrix<Scalar, 3, 1> x1;
    //     x1 << V.X(is[1], js[1]), V.Y(is[1], js[1]), V.Z(is[1], js[1]);
    //     x1 -= x0;
    //     Eigen::Matrix<Scalar, 3, 1> x2;
    //     x2 << V.X(is[2], js[2]), V.Y(is[2], js[2]), V.Z(is[2], js[2]);
    //     x2 -= x0;

    //     auto solve_d_new = [&] () -> Scalar {
    //         bool is_inf = false;
    //         bool all_inf = true;
    //         for (int k = 1; k < 3; ++k) {
    //             bool is_solve_dijkstra = false;
    //             if (std::isinf(D(is[k], js[k]))) {
    //                 is_inf = true;
    //             }
    //             else {
    //                 all_inf = false;
    //             }
    //         }
    //         if (all_inf) {
    //             return std::numeric_limits<Scalar>::infinity();
    //         }
    //         if (is_inf) {
    //             return solve_dijkstra(x1, x2, d1, d2);
    //         }
    //         Eigen::Matrix<Scalar, 2, 1> t;
    //         t << d1, d2;
    //         Eigen::Map<Eigen::Matrix<Scalar, 1, 2> > t_T(t.data());
    //         size_t coeff_idx = j_c + i_c * len;
    //         Scalar a = coeff.a[coeff_idx];
    //         Scalar b = coeff.b[coeff_idx] * t;
    //         Scalar c = t_T * coeff.c[coeff_idx] * t;
    //         Scalar d0 = solve_quadratic(a, -2.0 * b, c - 1.0);
    //         // if (std::isnan(d0)) std::cout << "n";
    //         // if (d0 < std::max(d1, d2)) std::cout << "m";
    //         if (std::isnan(d0) || d0 < std::max(d1, d2)) {
    //             return solve_dijkstra(x1, x2, d1, d2);
    //         }
    //         const Eigen::Matrix<Scalar, 2, 2> &Q = coeff.c[coeff_idx];
    //         Eigen::Matrix<Scalar, 2, 1> d0_1;
    //         d0_1 << d0, d0;
    //         Eigen::Matrix<Scalar, 2, 1> monotonicity_vec = Q * (t - d0_1);
    //         bool monotonicity_cond = (monotonicity_vec.array() > 0.0).any();
    //         if (monotonicity_cond) std::cout << "c";
    //         if (monotonicity_cond) {
    //             return solve_dijkstra(x1, x2, d1, d2);
    //         }
    //         assert(!(std::isnan(d0) || d0 == 1.0 / 0.0 && (d1 != 1.0 / 0.0 || d2 != 1.0 / 0.0)));
    //         // std::cout << "triangle = [(" << is[0] << "," << js[0] << "),(" << is[1] << "," << js[1] << "),(" << is[2] << "," << js[2] << ")], d0 = " << d0 << std::endl;
    //         std::cout << "t";
    //         return d0;
    //     };

    //     Scalar &d = D(is[0], js[0]);
    //     Scalar d_new = solve_d_new();
    //     // Scalar d_new = solve_dijkstra(x1, x2, d1, d2); // Test only the dijkstra part of the algorithm

    //     Scalar d_min = std::min(d, d_new);
    //     d = std::max(d_min, 0.0);
    // };

    // auto solve_upwards = [&] () {
    //     for (size_t i = 1; i < data.rows; ++i) {
    //         std::array<size_t, 3> is = { i, i - 1, i - 1 };
    //         size_t j = 0;
    //         auto right = [&] () {
    //             solve(data.upwards.right, data.V_row, D_row, is, std::array<size_t, 3>{ j, j, j + 1 }, i - 1, j, data.cols - 1);
    //         };
    //         auto left = [&] () {
    //             solve(data.upwards.left, data.V_row, D_row, is, std::array<size_t, 3>{ j, j - 1, j }, i - 1, j - 1, data.cols - 1);
    //         };
    //         right();
    //         for (j = 1; j < data.cols - 1; ++j) {
    //             left();
    //             right();
    //         }
    //         left();
    //     }
    // };
    // auto solve_downwards = [&] () {
    //     for (size_t i = 1; i < data.rows; ++i) {
    //         std::array<size_t, 3> is = { (data.rows - 1) - i, (data.rows - 1) - (i - 1), (data.rows - 1) - (i - 1) };
    //         size_t j = 0;
    //         auto left = [&] () {
    //             solve(data.downwards.right, data.V_row, D_row, is, std::array<size_t, 3>{ (data.cols - 1) - j, (data.cols - 1) - j, (data.cols - 1) - (j + 1) }, (data.rows - 1) - i, (data.cols - 1) - (j - 1), data.cols - 1);
    //         };
    //         auto right = [&] () {
    //             solve(data.downwards.left, data.V_row, D_row, is, std::array<size_t, 3>{ (data.cols - 1) - j, (data.cols - 1) - (j - 1), (data.cols - 1) - j }, (data.rows - 1) - i, (data.cols - 1) - j, data.cols - 1);
    //         };
    //         left();
    //         for (j = 1; j < data.cols - 1; ++j) {
    //             right();
    //             left();
    //         }
    //         right();
    //     }
    // };
    // auto solve_rightwards = [&] () {
    //     for (size_t j = 1; j < data.cols; ++j) {
    //         std::array<size_t, 3> js { j, j - 1, j - 1 };
    //         size_t i = 0;
    //         auto up = [&] () {
    //             solve(data.rightwards.left, data.V_col, D_col, std::array<size_t, 3>{ i, i + 1, i }, js, j - 1, i, data.rows - 1);
    //         };
    //         auto down = [&] () {
    //             solve(data.rightwards.right, data.V_col, D_col, std::array<size_t, 3>{ i, i, i -1 }, js, j - 1, i - 1, data.rows - 1);
    //         };
    //         up();
    //         for (i = 1; i < data.rows - 1; ++i) {
    //             down();
    //             up();
    //         }
    //         down();
    //     }
    // };
    // auto solve_leftwards = [&] () {
    //     for (size_t j = 1; j < data.cols; ++j) {
    //         std::array<size_t, 3> js = { (data.cols - 1) - j, (data.cols - 1) - (j - 1), (data.cols - 1) - (j - 1) };
    //         size_t i = 0;
    //         auto down = [&] () {
    //             solve(data.leftwards.left, data.V_col, D_col, std::array<size_t, 3>{ (data.rows - 1) - i, (data.rows - 1) - (i + 1), (data.rows - 1) - i }, js, (data.cols - 1) - j, (data.rows - 1) - (i - 1), data.rows - 1);
    //         };
    //         auto up = [&] () {
    //             solve(data.leftwards.right, data.V_col, D_col, std::array<size_t, 3>{ (data.rows - 1) - i, (data.rows - 1) - i, (data.rows - 1) - (i - 1) }, js, (data.cols - 1) - j, (data.rows - 1) - i, data.rows - 1);
    //         };
    //         down();
    //         for (i = 1; i < data.rows - 1; ++i) {
    //             up();
    //             down();
    //         }
    //         up();
    //     }
    // };

    // Resize and reset to infinity the distance vector
    for (size_t i = 0; i < rows * cols; ++i) {
        p_D[i] = std::numeric_limits<Scalar>::infinity();
    }
    // Set the source vertices as distance 0
    for (size_t s = 0; s < S.size(); ++s) {
        p_D[S[s]] = 0.0;
    }

    // Copy D to the device
    checkCuda(cudaMemcpy2D(d_D[0], d_D_pitch_bytes[0], p_D, cols * sizeof(Scalar), cols * sizeof(Scalar), rows, cudaMemcpyHostToDevice));

    for (size_t iter = 0; iter < N; ++iter) {
        // Upwards and downwards (row major)
        cudaEvent_t eventStart[4];
        cudaEvent_t eventStop[4];
        cudaStream_t stream[4];
        for (unsigned i = 0; i < 4; ++i) {
            checkCuda(cudaEventCreate(&eventStart[i]));
            checkCuda(cudaEventCreate(&eventStop[i]));
            checkCuda(cudaStreamCreate(&stream[i]));
        }

        // Upwards and downwards kernels
        {
            size_t tile_width = std::min(numWarps * warpSize, (size_t)maxThreads);
            size_t tile_height = omega + 1;
            size_t tile_offset = omega; // Every tile is shifted by 'Omega' (= 'tile_height - 1') to the left
            size_t overlap = std::max(2 * tile_offset, 2UL);
            size_t tile_eff_width = tile_width - overlap;
            size_t tile_eff_height = tile_height - 1;
            size_t tile_pitch = tile_width;
            size_t total_mem = tile_pitch * tile_height * sizeof(Scalar);
            size_t num_tiles = std::min(((cols + overlap) + (tile_eff_width - 1)) / tile_eff_width, (size_t)maxGridWidth);
            num_tiles = std::min(num_tiles, maxSharedMem / total_mem);
            // Define the dimensions
            dim3 blockDim(tile_width);
            dim3 gridDim(num_tiles);
            // Record the start of the kernel
            checkCuda(cudaEventRecord(eventStart[0], stream[0]));
            // Execute the kernel
            for (unsigned offset = 0; offset < rows - 1; offset += tile_eff_height) {
                solve_upwards<<<gridDim, blockDim, total_mem/*, stream[0]*/>>>(
                    d_D[0], V, d_C[0],
                    tile_width, tile_height, tile_pitch,
                    tile_eff_width, tile_eff_height, tile_offset, offset,
                    cols, rows,
                    d_D_pitch[0], d_C_pitch[0]
                );
            }
            // Record the end of the kernel
            checkCuda(cudaEventRecord(eventStop[0], stream[0]));
            // Record the start of the kernel
            checkCuda(cudaEventRecord(eventStart[1], stream[1]));
            // Execute the kernel
            for (unsigned offset = 0; offset < rows - 1; offset += tile_eff_height) {
                solve_downwards<<<gridDim, blockDim, total_mem/*, stream[1]*/>>>(
                    d_D[0], V, d_C[1],
                    tile_width, tile_height, tile_pitch,
                    tile_eff_width, tile_eff_height, tile_offset, offset,
                    cols, rows,
                    d_D_pitch[0], d_C_pitch[1]
                );
            }
            // Record the end of the kernel
            checkCuda(cudaEventRecord(eventStop[1], stream[1]));
        }
        // Wait for the kernels to finish
        checkCuda(cudaEventSynchronize(eventStop[0]));
        checkCuda(cudaEventSynchronize(eventStop[1]));
        // Copy transpose d_D[0] to d_D[1]
        {
            dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
            dim3 gridDim(
                (cols + BLOCK_DIM - 1) / BLOCK_DIM,
                (rows + BLOCK_DIM - 1) / BLOCK_DIM
            );
            transpose_kernel<<<gridDim, blockDim>>>(d_D[1], d_D_pitch[1], d_D[0], d_D_pitch[0], cols, rows);
        }
        // Rightwards and leftwards kernels
        {
            size_t tile_width = std::min(numWarps * warpSize, (size_t)maxThreads);
            size_t tile_height = omega + 1;
            size_t tile_offset = omega; // Every tile is shifted by 'Omega' (= 'tile_height - 1') to the left
            size_t overlap = std::max(2 * tile_offset, 2UL);
            size_t tile_eff_width = tile_width - overlap;
            size_t tile_eff_height = tile_height - 1;
            size_t tile_pitch = tile_width;
            size_t total_mem = tile_pitch * tile_height * sizeof(Scalar);
            size_t num_tiles = std::min(((rows + overlap) + (tile_eff_width - 1)) / tile_eff_width, (size_t)maxGridWidth);
            num_tiles = std::min(num_tiles, maxSharedMem / total_mem);
            // Define the dimensions
            dim3 blockDim(tile_width);
            dim3 gridDim(num_tiles);
            // Record the start of the kernel
            checkCuda(cudaEventRecord(eventStart[2], stream[2]));
            // Execute the kernel
            for (unsigned offset = 0; offset < cols - 1; offset += tile_eff_height) {
                solve_rightwards<<<gridDim, blockDim, total_mem/*, stream[2]*/>>>(
                    d_D[1], V, d_C[2],
                    tile_width, tile_height, tile_pitch,
                    tile_eff_width, tile_eff_height, tile_offset, offset,
                    rows, cols,
                    d_D_pitch[1], d_C_pitch[2]
                );
            }
            // Record the end of the kernel
            checkCuda(cudaEventRecord(eventStop[2], stream[2]));
            // Record the start of the kernel
            checkCuda(cudaEventRecord(eventStart[3], stream[3]));
            // Execute the kernel
            for (unsigned offset = 0; offset < cols - 1; offset += tile_eff_height) {
                solve_leftwards<<<gridDim, blockDim, total_mem/*, stream[3]*/>>>(
                    d_D[1], V, d_C[3],
                    tile_width, tile_height, tile_pitch,
                    tile_eff_width, tile_eff_height, tile_offset, offset,
                    rows, cols,
                    d_D_pitch[1], d_C_pitch[3]
                );
            }
            // Record the end of the kernel
            checkCuda(cudaEventRecord(eventStop[3], stream[3]));
        }
        // Wait for the kernels to finish
        checkCuda(cudaEventSynchronize(eventStop[2]));
        checkCuda(cudaEventSynchronize(eventStop[3]));
        // Copy transpose d_D[1] to d_D[0]
        {
            dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
            dim3 gridDim(
                (rows + BLOCK_DIM - 1) / BLOCK_DIM,
                (cols + BLOCK_DIM - 1) / BLOCK_DIM
            );
            transpose_kernel<<<gridDim, blockDim>>>(d_D[0], d_D_pitch[0], d_D[1], d_D_pitch[1], rows, cols);
        }
    }

    // Copy D from the device
    checkCuda(cudaMemcpy2D(p_D, cols * sizeof(Scalar), d_D[0], d_D_pitch_bytes[0], cols * sizeof(Scalar), rows, cudaMemcpyDeviceToHost));
}