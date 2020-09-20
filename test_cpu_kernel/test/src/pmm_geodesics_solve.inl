#include "pmm.h"

#include <Eigen/Dense>
#include <array>
#include <limits>
#include <cmath>
#include <assert.h>

template < typename Scalar, typename DerivedV, typename DerivedF, typename DerivedS, typename DerivedD>
PMM_INLINE void pmm_geodesics_solve(
    const PMMGeodesicsData<Scalar> & data,
    const Eigen::MatrixBase<DerivedV> & V,
    const Eigen::MatrixBase<DerivedF> & F,
    const Eigen::MatrixBase<DerivedS> & S,
    Eigen::PlainObjectBase<DerivedD> & D,
    size_t N
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
        Eigen::Matrix<Scalar, 3, 1> Mx1;
        Mx1 << V.X(is[1], js[1]), V.Y(is[1], js[1]), V.Z(is[1], js[1]);
        Mx1 -= x0;
        Eigen::Matrix<Scalar, 3, 1> Mx2;
        Mx2 << V.X(is[2], js[2]), V.Y(is[2], js[2]), V.Z(is[2], js[2]);
        Mx2 -= x0;

        auto solve_d_new = [&] () -> Scalar {

            size_t coeff_idx = j_c + i_c * len;

            Scalar a = coeff.a[coeff_idx];
            struct Scalar2 {
                Scalar x, y;
            };
            struct Scalar3 {
                Scalar x, y, z;
            };
            struct Scalar4 {
                Scalar x, y, z, w;
            };
            Scalar2 d = { d1, d2 };
            Scalar3 x1 = { Mx1.data()[0], Mx1.data()[1], Mx1.data()[2] };
            Scalar3 x2 = { Mx2.data()[0], Mx2.data()[1], Mx2.data()[2] };
            Scalar2 b2 = { coeff.b[coeff_idx].data()[0], coeff.b[coeff_idx].data()[1] };
            Scalar4 c4 = {
                coeff.c[coeff_idx].data()[0],
                coeff.c[coeff_idx].data()[1],
                coeff.c[coeff_idx].data()[2],
                coeff.c[coeff_idx].data()[3]
            };

            // The Scalar 2 b is a row vector
            // The Scalar 4 c is a col major 2x2 matrix
            const Scalar b = -2.0 * (b2.x * d.x + b2.y * d.y);
            const Scalar c = -1.0 + (c4.x * d.x + c4.y * d.y) * d.x + (c4.z * d.x + c4.w * d.y) * d.y;

            const Scalar a_2 = 2.0 * a;
            const Scalar sqroot = sqrt(b * b - 4.0 * a * c);
            const Scalar rhs = sqroot / a_2;
            const Scalar lhs = -b / a_2;
            const Scalar d_quadratic = std::max(lhs - rhs, lhs + rhs);

            const Scalar x1_norm = std::sqrt(x1.x * x1.x + x1.y * x1.y + x1.z * x1.z);
            const Scalar x2_norm = std::sqrt(x2.x * x2.x + x2.y * x2.y + x2.z * x2.z);
            const Scalar d_dijkstra = std::min(d.x + x1_norm, d.y + x2_norm);

            const bool cond_inf = std::isinf(d.x) || isinf(d.y);
            const bool cond_nan = std::isnan(d_quadratic);
            const bool cond_max = d_quadratic < std::max(d.x, d.y);
            const bool cond_monotonicity = (c4.x * (d.x - d_quadratic) + c4.z * (d.y - d_quadratic) > 0.0)
                                        || (c4.y * (d.x - d_quadratic) + c4.w * (d.y - d_quadratic) > 0.0);
            const bool cond = cond_nan || cond_max || cond_monotonicity || cond_inf;

            return cond ? d_dijkstra : d_quadratic;
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

    for (size_t iter = 0; iter < N; ++iter) {
        // Upwards and downwards (row major)
        solve_upwards();
        solve_downwards();
        D_col = D_row; // Update the column major distance map
        // Rightwards and leftwards (column major)
        solve_rightwards();
        solve_leftwards();
        D_row = D_col; // Update the row major distance map
    }

    // D_row is a mapping to D, so the distance map is already updated by the end of the loop
}