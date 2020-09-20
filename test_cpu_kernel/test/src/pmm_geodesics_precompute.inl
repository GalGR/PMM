#include "pmm.h"

#include <Eigen/Dense>

#include <array>

template < typename DerivedV, typename DerivedF, typename Scalar >
PMM_INLINE bool pmm_geodesics_precompute(
    size_t rows, size_t cols,
    const Eigen::MatrixBase<DerivedV> & V,
    const Eigen::MatrixBase<DerivedF> & F,
    PMMGeodesicsData<Scalar> & data,
    bool ignore_non_acute_triangles
) {
    data.resize(rows, cols);

    // Convert V to X, Y, Z
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data.V_row.X(i, j) = V(j + i * cols, 0);
            data.V_row.Y(i, j) = V(j + i * cols, 1);
            data.V_row.Z(i, j) = V(j + i * cols, 2);
        }
    }
    data.V_col.X = data.V_row.X;
    data.V_col.Y = data.V_row.Y;
    data.V_col.Z = data.V_row.Z;

    auto quadratic_coeff = [&] (
        std::array<Eigen::Matrix<Scalar, 3, 1>, 3> x,
        double &a,
        Eigen::Matrix<Scalar, 1, 2> &b,
        Eigen::Matrix<Scalar, 2, 2> &c
    ) {
        x[1] = x[1] - x[0];
        x[2] = x[2] - x[0];

        Eigen::Matrix<Scalar, 3, 2> X;
        X << x[1], x[2];
        Eigen::Matrix<Scalar, 2, 3> X_T = X.transpose();

        Eigen::Matrix<Scalar, 2, 2> E = X_T * X;

        // Assert e_12 >= 0
        if (!ignore_non_acute_triangles && E(0, 1) < 0) {
            throw std::runtime_error("The mesh contains non acute triangle");
        }

        c = E.inverse(); // Q == E^-1
        Eigen::Matrix<Scalar, 1, 2> ones_T = Eigen::Matrix<Scalar, 1, 2>::Ones();
        b = ones_T * c; // 1^T Q
        Eigen::Matrix<Scalar, 2, 1> ones = Eigen::Matrix<Scalar, 2, 1>::Ones();
        a = b * ones; // 1^T Q 1
    };

    auto quadratic_func = [&] (
        auto &tri_data,
        const std::array<size_t, 3> &is, const std::array<size_t, 3> &js,
        size_t i_c, size_t j_c, size_t len
    ) {
        size_t idx[3];
        for (int k = 0; k < 3; ++k) {
            idx[k] = js[k] + is[k] * cols;
        }
        size_t coeff_idx = j_c + i_c * len;
        std::array<Eigen::Matrix<Scalar, 3, 1>, 3> x;
        for (int k = 0; k < 3; ++k) {
            x[k] = V.row(idx[k]).transpose();
        }
        auto &a = tri_data.a[coeff_idx];
        auto &b = tri_data.b[coeff_idx];
        auto &c = tri_data.c[coeff_idx];
        quadratic_coeff(x, a, b, c);
    };

    auto upwards_right = [&] (size_t i, size_t j) {
        std::array<size_t, 3> is = { i, i - 1, i - 1 };
        std::array<size_t, 3> js = { j, j, j + 1 };
        quadratic_func(data.upwards.right, is, js, is[0] - 1, js[0], cols - 1);
    };
    auto upwards_left = [&] (size_t i, size_t j) {
        std::array<size_t, 3> is = { i, i - 1, i - 1 };
        std::array<size_t, 3> js = { j, j - 1, j };
        quadratic_func(data.upwards.left, is, js, is[0] - 1, js[0] - 1, cols - 1);
    };
    auto downwards_right = [&] (size_t i, size_t j) {
        std::array<size_t, 3> is = { i, i + 1, i + 1 };
        std::array<size_t, 3> js = { j, j + 1, j };
        quadratic_func(data.downwards.left, is, js, is[0], js[0], cols - 1);
    };
    auto downwards_left = [&] (size_t i, size_t j) {
        std::array<size_t, 3> is = { i, i + 1, i + 1 };
        std::array<size_t, 3> js = { j, j, j - 1 };
        quadratic_func(data.downwards.right, is, js, is[0], js[0] - 1, cols - 1);
    };
    auto rightwards_up = [&] (size_t i, size_t j) {
        std::array<size_t, 3> is = { i, i + 1, i};
        std::array<size_t, 3> js = { j, j - 1, j - 1 };
        quadratic_func(data.rightwards.left, is, js, js[0] - 1, is[0], rows - 1);
    };
    auto rightwards_down = [&] (size_t i, size_t j) {
        std::array<size_t, 3> is = { i, i, i - 1 };
        std::array<size_t, 3> js = { j, j - 1, j - 1 };
        quadratic_func(data.rightwards.right, is, js, js[0] - 1, is[0] - 1, rows - 1);
    };
    auto leftwards_up = [&] (size_t i, size_t j) {
        std::array<size_t, 3> is = { i, i, i + 1 };
        std::array<size_t, 3> js = { j, j + 1, j + 1 };
        quadratic_func(data.leftwards.right, is, js, js[0], is[0], rows - 1);
    };
    auto leftwards_down = [&] (size_t i, size_t j) {
        std::array<size_t, 3> is = { i, i - 1, i };
        std::array<size_t, 3> js = { j, j + 1, j + 1 };
        quadratic_func(data.leftwards.left, is, js, js[0], is[0] - 1, rows - 1);
    };

    // Calculate the precalculations
    try {
        for (size_t i = 1; i < rows; ++i) {
            upwards_right(i, 0);
            for (size_t j = 1; j < cols - 1; ++j) {
                upwards_left(i, j);
                upwards_right(i, j);
            }
            upwards_left(i, cols - 1);
        }
        for (size_t i = 0; i < rows - 1; ++i) {
            downwards_right(i, 0);
            for (size_t j = 1; j < cols - 1; ++j) {
                downwards_left(i, j);
                downwards_right(i, j);
            }
            downwards_left(i, cols - 1);
        }
        for (size_t j = 1; j < cols; ++j) {
            rightwards_up(0, j);
        }
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols; ++j) {
                rightwards_down(i, j);
            }
            for (size_t j = 1; j < cols; ++j) {
                rightwards_up(i, j);
            }
        }
        for (size_t j = 1; j < cols; ++j) {
            rightwards_down(rows - 1, j);
        }
        for (size_t j = 0; j < cols - 1; ++j) {
            leftwards_up(0, j);
        }
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 0; j < cols - 1; ++j) {
                leftwards_down(i, j);
            }
            for (size_t j = 0; j < cols - 1; ++j) {
                leftwards_up(i, j);
            }
        }
        for (size_t j = 0; j < cols - 1; ++j) {
            leftwards_down(rows - 1, j);
        }
    }
    catch (const std::runtime_error &e) {
        return false;
    }

    return true;
}