#include "pmm.cuh"

#include <Eigen/Dense>
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

__device__ Scalar solve(const Scalar3 x1, const Scalar3 x2, const Scalar2 d, const Scalar a, const Scalar2 b2, const Scalar4 c4) {
    // The Scalar 2 b is a row vector
    // The Scalar 4 c is a col major 2x2 matrix
    const Scalar b = -2.0 * b2.x * d.x + b2.y * d.y;
    const Scalar c = -1.0 + (c4.x * d.x + c4.y * d.y) * d.x + (c4.z * d.x + c4.w * d.y) * d.y;

    const Scalar a_2 = 2.0 * a;
    const Scalar sqroot = sqrt(b * b - 4.0 * a * c);
    const Scalar rhs = sqroot / a_2;
    const Scalar lhs = -b / a_2;
    const Scalar d_quadratic = fmax(lhs - rhs, lhs + rhs);

    const Scalar x1_norm = sqrt(x1.x * x1.x + x1.y * x1.y + x1.z * x1.z);
    const Scalar x2_norm = sqrt(x2.x * x2.x + x2.y * x2.y + x2.z * x2.z);
    const Scalar d_dijkstra = fmin(d.x + x1_norm, d.y + x2_norm);

    const bool cond_max = d_quadratic < fmax(d.x, d.y);
    const bool cond_monotonicity = ((c4.x * d.x + c4.z * d.y) > d_quadratic) * ((c4.y * d.x + c4.w * d.y) > d_quadratic);
    const bool cond = cond_max * cond_monotonicity;

    return d_quadratic * !cond + d_dijkstra * cond;
}

__global__ void solve_upwards(Scalar *D, cudaTextureObject_t V, Scalar *C, unsigned tile_width, unsigned tile_height, unsigned tile_pitch, unsigned width, unsigned height, unsigned D_pitch, unsigned C_pitch) {
    extern __shared__ Scalar d_shared[];
    // How many elements we go over each stride
    const unsigned stride = gridDim.x * (blockDim.x - 2 * (tile_height - 2));
    // Stride all over the matrix
    for (unsigned offset = 0; offset < (width * (height - 1) / (tile_height - 1)) + stride - 1; offset += stride) {
        // Every tile is shifted by 'Omega - 1' (= 'tile_height - 2') to the left
        const unsigned x = offset % width + blockIdx.x * (tile_width - 2 * (tile_height - 2)) + threadIdx.x - (tile_height - 2); // Negative number will result in x > width
        const unsigned y = (offset / width) * (tile_height - 1);
        // Copy for each tile, 'D' to shared memory 'd_shared'
        for (unsigned i = 0; i < tile_height && x < width; ++i) {
            if (y + i < height) {
                d_shared[threadIdx.x + i * tile_pitch] = D[x + (y + i) * D_pitch];
            }
        }
        // Sync all threads in block after copying to shared memory
        __syncthreads();
        // C is organized as such: right, (left, right), left
        for (unsigned i = 1; i < tile_height && x < width; ++i) {
            const unsigned idx_d_0 = (threadIdx.x + 0) + (i + 0) * tile_pitch;
            Scalar d_l = d_shared[(threadIdx.x - 1) + (i - 1) * tile_pitch];
            Scalar d_m = d_shared[(threadIdx.x + 0) + (i - 1) * tile_pitch];
            Scalar d_r = d_shared[(threadIdx.x + 1) + (i - 1) * tile_pitch];
            TexScalar v_x_l = tex3D<TexScalar>(V, (x - 1) + 0.5, (y + i - 1) + 0.5, 0 + 0.5);
            TexScalar v_x_m = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i - 1) + 0.5, 0 + 0.5);
            TexScalar v_x_r = tex3D<TexScalar>(V, (x + 1) + 0.5, (y + i - 1) + 0.5, 0 + 0.5);
            TexScalar v_x_0 = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i + 0) + 0.5, 0 + 0.5);
            TexScalar v_y_l = tex3D<TexScalar>(V, (x - 1) + 0.5, (y + i - 1) + 0.5, 1 + 0.5);
            TexScalar v_y_m = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i - 1) + 0.5, 1 + 0.5);
            TexScalar v_y_r = tex3D<TexScalar>(V, (x + 1) + 0.5, (y + i - 1) + 0.5, 1 + 0.5);
            TexScalar v_y_0 = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i + 0) + 0.5, 1 + 0.5);
            TexScalar v_z_l = tex3D<TexScalar>(V, (x - 1) + 0.5, (y + i - 1) + 0.5, 2 + 0.5);
            TexScalar v_z_m = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i - 1) + 0.5, 2 + 0.5);
            TexScalar v_z_r = tex3D<TexScalar>(V, (x + 1) + 0.5, (y + i - 1) + 0.5, 2 + 0.5);
            TexScalar v_z_0 = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i + 0) + 0.5, 2 + 0.5);
            Scalar3 x_l = make_Scalar3(v_x_l - v_x_0, v_y_l - v_y_0, v_z_l - v_z_0);
            Scalar3 x_m = make_Scalar3(v_x_m - v_x_0, v_y_m - v_y_0, v_z_m - v_z_0);
            Scalar3 x_r = make_Scalar3(v_x_r - v_x_0, v_y_r - v_y_0, v_z_r - v_z_0);
            if (x < width && y < height) {
                if (x > 0) { // Left triangle -- even rows
                    // Scalar  a  = *reinterpret_cast<Scalar *>(&C[(PMM_A_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (y + i) * C_pitch * 2]);
                    // Scalar2 b  = *reinterpret_cast<Scalar2*>(&C[(PMM_B_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (y + i) * C_pitch * 2]);
                    Scalar4 ab = *reinterpret_cast<Scalar4*>(&C[(PMM_A_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (y + i) * C_pitch * 2]);
                    Scalar4 c  = *reinterpret_cast<Scalar4*>(&C[(PMM_C_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (y + i) * C_pitch * 2]);
                    // Scalar d_new = solve(x_l, x_m, d_l, d_m, a, b, c);
                    Scalar d_new = solve(x_l, x_m, make_Scalar2(d_l, d_m), ab.x, make_Scalar2(ab.z, ab.w), c);
                    d_shared[idx_d_0] = fmin(d_new, d_shared[idx_d_0]);
                }
                if (x < width - 1) { // Right triangle -- odd rows
                    // Scalar  a  = *reinterpret_cast<Scalar *>(&C[(PMM_A_OFF + 0) + x * PMM_COEFF_PITCH + (y + i) * C_pitch * 2 + C_pitch]);
                    // Scalar2 b  = *reinterpret_cast<Scalar2*>(&C[(PMM_B_OFF + 0) + x * PMM_COEFF_PITCH + (y + i) * C_pitch * 2 + C_pitch]);
                    Scalar4 ab = *reinterpret_cast<Scalar4*>(&C[(PMM_A_OFF + 0) + x * PMM_COEFF_PITCH + (y + i) * C_pitch * 2 + C_pitch]);
                    Scalar4 c  = *reinterpret_cast<Scalar4*>(&C[(PMM_C_OFF + 0) + x * PMM_COEFF_PITCH + (y + i) * C_pitch * 2 + C_pitch]);
                    // Scalar d_new = solve(x_m, x_r, d_m, d_r, a, b, c);
                    Scalar d_new = solve(x_m, x_r, make_Scalar2(d_m, d_r), ab.x, make_Scalar2(ab.z, ab.w), c);
                    d_shared[idx_d_0] = fmin(d_new, d_shared[idx_d_0]);
                }
            }
        }
        // Sync all threads after they calculated the distances in shared memory
        __syncthreads();
        // Copy for each tile, 'd_shared' to global memory 'D'
        for (unsigned i = 1; i < tile_height && x < width; ++i) {
            if (y + i < height && threadIdx.x >= (i - 1) && threadIdx.x < (i - 1) + (tile_width - 2 * (tile_height - 2))) {
                D[x + (y + i) * D_pitch] = d_shared[threadIdx.x + i * tile_pitch];
            }
        }
    }
}

template <typename Scalar, typename DerivedS, typename DerivedD>
PMM_INLINE void pmm_geodesics_solve(
    size_t rows, size_t cols,
    int maxGridWidth,
    int maxThreads,
    int warpSize,
    size_t maxSharedMem,
    std::array<Scalar*, 4> &d_C,
    const std::array<size_t, 4> &d_C_pitch_bytes,
    const std::array<size_t, 4> &d_C_pitch,
    const cudaTextureObject_t V,
    const Eigen::MatrixBase<DerivedS> &S,
    Eigen::MatrixBase<DerivedD> &D,
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
    D.setConstant(std::numeric_limits<Scalar>::infinity());
    // Set the source vertices as distance 0
    for (size_t s = 0; s < S.size(); ++s) {
        D(S(s)) = 0;
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

        {
            size_t tile_width = std::min(numWarps * warpSize, (size_t)maxThreads);
            size_t tile_height = omega + 1;
            size_t total_mem = tile_width * tile_height * sizeof(Scalar);
            size_t num_tiles = std::min(((cols + tile_height - 2) + (tile_width - 2 * (tile_height - 2) - 1)) / (tile_width - 2 * (tile_height - 2)), (size_t)maxGridWidth);
            num_tiles = std::min(num_tiles, maxSharedMem / total_mem);
            // Define the dimensions
            dim3 blockDim(tile_width);
            dim3 gridDim(num_tiles);
            // Record the start of the kernel
            checkCuda(cudaEventRecord(eventStart[0], stream[0]));
            // Execute the kernel
            solve_upwards<<<blockDim, gridDim, total_mem, stream[0]>>>(
                d_D[0], V, d_C[0],
                tile_width, tile_height, tile_width,
                cols, rows,
                d_D_pitch[0], d_C_pitch[0]
            );
            // Record the end of the kernel
            checkCuda(cudaEventRecord(eventStop[0], stream[0]));
            // solve_downwards(downwardsEvent);
        }
        // Wait for the kernels to finish
        checkCuda(cudaEventSynchronize(eventStop[0]));
        // checkCuda(cudaEventSynchronize(eventStop[1]));
        // // D_col = D_row; // Update the column major distance map
        // copy_transpose_D();
        // // Rightwards and leftwards (column major)
        // solve_rightwards();
        // solve_leftwards();
        // // Wait for the kernels to finish
        // checkCuda(cudaEventSynchronize(rightwardsEvent));
        // checkCuda(cudaEventSynchronize(leftwardsEvent));
        // // D_row = D_col; // Update the row major distance map
        // copy_transpose_D_T();
    }

    // Copy D from the device
    checkCuda(cudaMemcpy2D(p_D, cols * sizeof(Scalar), d_D[0], d_D_pitch[0], cols * sizeof(Scalar), rows, cudaMemcpyDeviceToHost));

    // DEBUG TEST: replace infinity with 0
    D = (D.array() == std::numeric_limits<Scalar>::infinity()).select(0, D);
}