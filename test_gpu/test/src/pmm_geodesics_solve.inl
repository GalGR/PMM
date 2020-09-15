#include "pmm.h"

#include <Eigen/Dense>
#include <array>
#include <limits>
#include <cmath>
#include <assert.h>
#include "utils_cuda.h"

enum PMMCudaStreams {
    PMM_UPWARDS_STREAM = 1,
    PMM_DOWNWARDS_STREAM,
    PMM_RIGHTWARDS_STREAM,
    PMM_LEFTWARDS_STREAM,
};

#ifndef SCALAR_
    #define SCALAR_
    #if defined(SCALAR_DOUBLE)
        typedef double Scalar;
        typedef double2 Scalar2;
        typedef double3 Scalar3;
        typedef double4 Scalar4;
    #else
        typedef float Scalar;
        typedef float2 Scalar2;
        typedef float3 Scalar3;
        typedef float4 Scalar4;
    #endif
#endif

__global__ void solve_upwards(Scalar *D, cudaTextureObject_t V, Scalar *C, unsigned tile_width, unsigned tile_height, unsigned tile_pitch, unsigned width, unsigned height, unsigned D_pitch, unsigned V_pitch, unsigned C_pitch) {
    extern __shared__ Scalar d_shared[];
    // How many elements we go over each stride
    int stride = gridDim.x * blockDim.x;
    // Stride all over the matrix
    for (int offset = 0; offset < (width * height / (tile_height - 1)) + stride - 1; offset += stride) {
        // Every tile is shifted by Omega (= tile_height) to the left
        int x = offset % width + blockIdx.x * tile_width + threadIdx.x - tile_height;
        int y = (offset / width) * (tile_height - 1);
        // Copy for each tile, 'D' to shared memory 'd_shared'
        for (int i = 0; i < tile_height && x >= 0 && x < width; ++i) {
            if (y + i < height) {
                d_shared[threadIdx.x + i * tile_pitch] = D[x + (y + i) * D_pitch];
            }
        }
        // Sync all threads in block after copying to shared memory
        __syncthreads();
        // C is organized as such: right, (left, right), left
        for (int i = 1; i < tile_height && x >= 0 && x < width; ++i) {
            const int idx_d_0 = (threadIdx.x + 0) + (i + 0) * tile_pitch;
            Scalar d_l = d[(threadIdx.x - 1) + (i - 1) * tile_pitch];
            Scalar d_m = d[(threadIdx.x + 0) + (i - 1) * tile_pitch];
            Scalar d_r = d[(threadIdx.x + 1) + (i - 1) * tile_pitch];
            Scalar v_x_l = tex3D<Scalar>(V, (x - 1) + 0.5, (y + i - 1) + 0.5, 0 + 0.5);
            Scalar v_x_m = tex3D<Scalar>(V, (x + 0) + 0.5, (y + i - 1) + 0.5, 0 + 0.5);
            Scalar v_x_r = tex3D<Scalar>(V, (x + 1) + 0.5, (y + i - 1) + 0.5, 0 + 0.5);
            Scalar v_x_0 = tex3D<Scalar>(V, (x + 0) + 0.5, (y + i + 0) + 0.5, 0 + 0.5);
            Scalar v_y_l = tex3D<Scalar>(V, (x - 1) + 0.5, (y + i - 1) + 0.5, 1 + 0.5);
            Scalar v_y_m = tex3D<Scalar>(V, (x + 0) + 0.5, (y + i - 1) + 0.5, 1 + 0.5);
            Scalar v_y_r = tex3D<Scalar>(V, (x + 1) + 0.5, (y + i - 1) + 0.5, 1 + 0.5);
            Scalar v_y_0 = tex3D<Scalar>(V, (x + 0) + 0.5, (y + i + 0) + 0.5, 1 + 0.5);
            Scalar v_z_l = tex3D<Scalar>(V, (x - 1) + 0.5, (y + i - 1) + 0.5, 2 + 0.5);
            Scalar v_z_m = tex3D<Scalar>(V, (x + 0) + 0.5, (y + i - 1) + 0.5, 2 + 0.5);
            Scalar v_z_r = tex3D<Scalar>(V, (x + 1) + 0.5, (y + i - 1) + 0.5, 2 + 0.5);
            Scalar v_z_0 = tex3D<Scalar>(V, (x + 0) + 0.5, (y + i + 0) + 0.5, 2 + 0.5);
            Scalar3 x_l = { v_x_l - v_x_0, v_y_l - v_y_0, v_z_l - v_z_0 };
            Scalar3 x_m = { v_x_m - v_x_0, v_y_m - v_y_0, v_z_m - v_z_0 };
            Scalar3 x_r = { v_x_r - v_x_0, v_y_r - v_y_0, v_z_r - v_z_0 };
            if (x < width && y < height) {
                if (x > 0) { // Left triangle
                    Scalar a  = *(Scalar *)(&C[(PMM_A_OFF + 0) + x * PMM_COEFF_PITCH * 2 - PMM_COEFF_PITCH + (y + i) * C_pitch]);
                    Scalar2 b = *(Scalar2*)(&C[(PMM_B_OFF + 0) + x * PMM_COEFF_PITCH * 2 - PMM_COEFF_PITCH + (y + i) * C_pitch]);
                    Scalar4 c = *(Scalar4*)(&C[(PMM_C_OFF + 0) + x * PMM_COEFF_PITCH * 2 - PMM_COEFF_PITCH + (y + i) * C_pitch]);
                    Scalar d_new = solve(x_l, x_m, d_l, d_m, a, b, c);
                    d[idx_d_0] = min(d_new, d[idx_d_0]);
                }
                if (x < width - 1) { // Right triangle
                    Scalar a  = *(Scalar *)(&C[(PMM_A_OFF + 0) + x * PMM_COEFF_PITCH * 2 + (y + i) * C_pitch]);
                    Scalar2 b = *(Scalar2*)(&C[(PMM_B_OFF + 0) + x * PMM_COEFF_PITCH * 2 + (y + i) * C_pitch]);
                    Scalar4 c = *(Scalar4*)(&C[(PMM_C_OFF + 0) + x * PMM_COEFF_PITCH * 2 + (y + i) * C_pitch]);
                    Scalar d_new = solve(x_m, x_r, d_m, d_r, a, b, c);
                    d[idx_d_0] = min(d_new, d[idx_d_0]);
                }
            }
        }
        // Sync all threads after they calculated the distances in shared memory
        __syncthreads();
        // Copy for each tile, 'd_shared' to global memory 'D'
        for (int i = 0; i < tile_height && x >= 0 && x < width; ++i) {
            if (y + i < height) {
                D[x + (y + i) * D_pitch] = d_shared[threadIdx.x + i * tile_pitch];
            }
        }
    }
}

template < typename Scalar, typename DerivedV, typename DerivedF, typename DerivedS, typename DerivedD>
PMM_INLINE void pmm_geodesics_solve(
    const PMMCoeffs<Scalar> &coeffs,
    std::array<Scalar*, PMM_ARRAYS> &d_coeffs,
    Scalar *p_coeffs,
    const Eigen::MatrixBase<DerivedV> & V,
    const Eigen::MatrixBase<DerivedF> & F,
    const Eigen::MatrixBase<DerivedS> & S,
    Eigen::PlainObjectBase<DerivedD> & D,
    size_t N,
    bool D_empty
) {
    // Resize and reset to infinity the distance vector
    D.setConstant(data.rows * data.cols, 1, std::numeric_limits<Scalar>::infinity());
    // Set the source vertices as distance 0
    for (size_t s = 0; s < S.size(); ++s) {
        D(S(s)) = 0;
    }

    // Create a D_row (row major) and D_col (column major) distance maps
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > D_row(D.data(), data.rows, data.cols);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> D_col = D_row;

    auto solve = [&] (
        const auto &coeff,
        const auto &V,
        auto &D,
        const std::array<size_t, 3> &is,
        const std::array<size_t, 3> &js,
        size_t i_c, size_t j_c, size_t len
    ) {

        auto solve_quadratic = [&] (Scalar a, Scalar b, Scalar c) -> Scalar {
            Scalar a_2 = 2 * a;
            Scalar sqroot = std::sqrt(b * b - 4 * a * c);
            Scalar rhs = sqroot / a_2;
            Scalar lhs = -b / a_2;
            Scalar res = std::max(lhs - rhs, lhs + rhs);
            // assert(!(std::isnan(res) || res == 1.0 / 0.0));
            return res;
        };

        auto solve_dijkstra = [&] (const Eigen::Matrix<Scalar, 3, 1> &x1, const Eigen::Matrix<Scalar, 3, 1> &x2, Scalar d1, Scalar d2) -> Scalar {
            return std::min(d1 + x1.norm(), d2 + x2.norm());
        };

        Scalar d1 = D(is[1], js[1]);
        Scalar d2 = D(is[2], js[2]);

        Eigen::Matrix<Scalar, 3, 1> x0;
        x0 << V.X(is[0], js[0]), V.Y(is[0], js[0]), V.Z(is[0], js[0]);
        Eigen::Matrix<Scalar, 3, 1> x1;
        x1 << V.X(is[1], js[1]), V.Y(is[1], js[1]), V.Z(is[1], js[1]);
        x1 -= x0;
        Eigen::Matrix<Scalar, 3, 1> x2;
        x2 << V.X(is[2], js[2]), V.Y(is[2], js[2]), V.Z(is[2], js[2]);
        x2 -= x0;

        auto solve_d_new = [&] () -> Scalar {
            bool is_inf = false;
            bool all_inf = true;
            for (int k = 1; k < 3; ++k) {
                bool is_solve_dijkstra = false;
                if (std::isinf(D(is[k], js[k]))) {
                    is_inf = true;
                }
                else {
                    all_inf = false;
                }
            }
            if (all_inf) {
                return std::numeric_limits<Scalar>::infinity();
            }
            if (is_inf) {
                return solve_dijkstra(x1, x2, d1, d2);
            }
            Eigen::Matrix<Scalar, 2, 1> t;
            t << d1, d2;
            Eigen::Map<Eigen::Matrix<Scalar, 1, 2> > t_T(t.data());
            size_t coeff_idx = j_c + i_c * len;
            Scalar a = coeff.a[coeff_idx];
            Scalar b = coeff.b[coeff_idx] * t;
            Scalar c = t_T * coeff.c[coeff_idx] * t;
            Scalar d0 = solve_quadratic(a, -2.0 * b, c - 1.0);
            // if (std::isnan(d0)) std::cout << "n";
            // if (d0 < std::max(d1, d2)) std::cout << "m";
            if (std::isnan(d0) || d0 < std::max(d1, d2)) {
                return solve_dijkstra(x1, x2, d1, d2);
            }
            const Eigen::Matrix<Scalar, 2, 2> &Q = coeff.c[coeff_idx];
            Eigen::Matrix<Scalar, 2, 1> d0_1;
            d0_1 << d0, d0;
            Eigen::Matrix<Scalar, 2, 1> monotonicity_vec = Q * (t - d0_1);
            bool monotonicity_cond = (monotonicity_vec.array() > 0.0).any();
            if (monotonicity_cond) std::cout << "c";
            if (monotonicity_cond) {
                return solve_dijkstra(x1, x2, d1, d2);
            }
            assert(!(std::isnan(d0) || d0 == 1.0 / 0.0 && (d1 != 1.0 / 0.0 || d2 != 1.0 / 0.0)));
            // std::cout << "triangle = [(" << is[0] << "," << js[0] << "),(" << is[1] << "," << js[1] << "),(" << is[2] << "," << js[2] << ")], d0 = " << d0 << std::endl;
            std::cout << "t";
            return d0;
        };

        Scalar &d = D(is[0], js[0]);
        Scalar d_new = solve_d_new();
        // Scalar d_new = solve_dijkstra(x1, x2, d1, d2); // Test only the dijkstra part of the algorithm

        Scalar d_min = std::min(d, d_new);
        d = std::max(d_min, 0.0);
    };

    auto solve_upwards = [&] () {
        for (size_t i = 1; i < data.rows; ++i) {
            std::array<size_t, 3> is = { i, i - 1, i - 1 };
            size_t j = 0;
            auto right = [&] () {
                solve(data.upwards.right, data.V_row, D_row, is, std::array<size_t, 3>{ j, j, j + 1 }, i - 1, j, data.cols - 1);
            };
            auto left = [&] () {
                solve(data.upwards.left, data.V_row, D_row, is, std::array<size_t, 3>{ j, j - 1, j }, i - 1, j - 1, data.cols - 1);
            };
            right();
            for (j = 1; j < data.cols - 1; ++j) {
                left();
                right();
            }
            left();
        }
    };
    auto solve_downwards = [&] () {
        for (size_t i = 1; i < data.rows; ++i) {
            std::array<size_t, 3> is = { (data.rows - 1) - i, (data.rows - 1) - (i - 1), (data.rows - 1) - (i - 1) };
            size_t j = 0;
            auto left = [&] () {
                solve(data.downwards.right, data.V_row, D_row, is, std::array<size_t, 3>{ (data.cols - 1) - j, (data.cols - 1) - j, (data.cols - 1) - (j + 1) }, (data.rows - 1) - i, (data.cols - 1) - (j - 1), data.cols - 1);
            };
            auto right = [&] () {
                solve(data.downwards.left, data.V_row, D_row, is, std::array<size_t, 3>{ (data.cols - 1) - j, (data.cols - 1) - (j - 1), (data.cols - 1) - j }, (data.rows - 1) - i, (data.cols - 1) - j, data.cols - 1);
            };
            left();
            for (j = 1; j < data.cols - 1; ++j) {
                right();
                left();
            }
            right();
        }
    };
    auto solve_rightwards = [&] () {
        for (size_t j = 1; j < data.cols; ++j) {
            std::array<size_t, 3> js { j, j - 1, j - 1 };
            size_t i = 0;
            auto up = [&] () {
                solve(data.rightwards.left, data.V_col, D_col, std::array<size_t, 3>{ i, i + 1, i }, js, j - 1, i, data.rows - 1);
            };
            auto down = [&] () {
                solve(data.rightwards.right, data.V_col, D_col, std::array<size_t, 3>{ i, i, i -1 }, js, j - 1, i - 1, data.rows - 1);
            };
            up();
            for (i = 1; i < data.rows - 1; ++i) {
                down();
                up();
            }
            down();
        }
    };
    auto solve_leftwards = [&] () {
        for (size_t j = 1; j < data.cols; ++j) {
            std::array<size_t, 3> js = { (data.cols - 1) - j, (data.cols - 1) - (j - 1), (data.cols - 1) - (j - 1) };
            size_t i = 0;
            auto down = [&] () {
                solve(data.leftwards.left, data.V_col, D_col, std::array<size_t, 3>{ (data.rows - 1) - i, (data.rows - 1) - (i + 1), (data.rows - 1) - i }, js, (data.cols - 1) - j, (data.rows - 1) - (i - 1), data.rows - 1);
            };
            auto up = [&] () {
                solve(data.leftwards.right, data.V_col, D_col, std::array<size_t, 3>{ (data.rows - 1) - i, (data.rows - 1) - i, (data.rows - 1) - (i - 1) }, js, (data.cols - 1) - j, (data.rows - 1) - i, data.rows - 1);
            };
            down();
            for (i = 1; i < data.rows - 1; ++i) {
                up();
                down();
            }
            up();
        }
    };

    // Create pinned memory allocation for 

    // Copy D to the device
    copy_host_to_device_D();

    for (size_t iter = 0; iter < N; ++iter) {
        // Upwards and downwards (row major)
        cudaEvent_t upwardsEvent;
        cudaEvent_t downwardsEvent;
        cudaEvent_t rightwardsEvent;
        cudaEvent_t leftwardsEvent;
        checkCuda(cudaEventCreate(&upwardsEvent));
        checkCuda(cudaEventCreate(&downwardsEvent));
        checkCuda(cudaEventCreate(&rightwardsEvent));
        checkCuda(cudaEventCreate(&leftwardsEvent));

        solve_upwards(upwardsEvent);
        solve_downwards(downwardsEvent);
        // Wait for the kernels to finish
        checkCuda(cudaEventSynchronize(upwardsEvent));
        checkCuda(cudaEventSynchronize(downwardsEvent));
        // D_col = D_row; // Update the column major distance map
        copy_transpose_D();
        // Rightwards and leftwards (column major)
        solve_rightwards();
        solve_leftwards();
        // Wait for the kernels to finish
        checkCuda(cudaEventSynchronize(rightwardsEvent));
        checkCuda(cudaEventSynchronize(leftwardsEvent));
        // D_row = D_col; // Update the row major distance map
        copy_transpose_D_T();
    }

    copy_device_to_host_D();
}