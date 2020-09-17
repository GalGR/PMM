#include "pmm.cuh"

#include <Eigen/Dense>

#include <array>
#include <cstring>

template <typename Scalar, typename DerivedV>
PMM_INLINE bool pmm_geodesics_precompute(
    size_t rows, size_t cols,
    const Eigen::MatrixBase<DerivedV> &V,
    std::array<std::vector<Scalar>, 4> &C,
    bool ignore_non_acute_triangles
) {
    PMMGeodesicsData<Scalar> data;
    return pmm_geodesics_precompute(data,rows, cols, V, C, ignore_non_acute_triangles);
}

template <typename Scalar, typename DerivedV>
PMM_INLINE bool pmm_geodesics_precompute(
    PMMGeodesicsData<Scalar> &data,
    size_t rows, size_t cols,
    const Eigen::MatrixBase<DerivedV> &V,
    std::array<std::vector<Scalar>, 4> &C,
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
        Scalar &a,
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

    // Convert PMMGeodesicsData format to PMMCoeffs format
    // for (size_t dir = 0; dir < 2; ++dir) {
    //     for (size_t tri = 0; tri < 2; ++tri) {
    //         coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_A_OFF].resize((rows - 1) * (cols - 1));
    //         for (unsigned k = 0; k < 2; ++k) {
    //             coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_B_OFF + k].resize((rows - 1) * (cols - 1));
    //         }
    //         for (unsigned k = 2; k < 4; ++k) {
    //             coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_C_OFF + k].resize((rows - 1) * (cols - 1));
    //         }
    //         for (size_t i = 0; i < rows - 1; ++i) {
    //             for (size_t j = 0; j < cols - 1; ++j) {
    //                 // Copy from A to A
    //                 coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_A_OFF][j + i * (cols - 1)] = data[dir][tri].a[j + i * (cols - 1)];
    //                 // Copy from B to B
    //                 for (unsigned k = 0; k < 2; ++k) {
    //                     coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_B_OFF + k][j + i * (cols - 1)] = data[dir][tri].b[j + i * (cols - 1)].data()[k];
    //                 }
    //                 // Copy from C to C
    //                 for (unsigned k = 0; k < 4; ++k) {
    //                     coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_C_OFF + k][j + i * (cols - 1)] = data[dir][tri].c[j + i * (cols - 1)].data()[k];
    //                 }
    //             }
    //         }
    //     }
    // }
    // for (size_t dir = 2; dir < 4; ++dir) {
    //     for (size_t tri = 0; tri < 2; ++tri) {
    //         coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_A_OFF].resize((cols - 1) * (rows - 1));
    //         for (unsigned k = 0; k < 2; ++k) {
    //             coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_B_OFF + k].resize((cols - 1) * (rows - 1));
    //         }
    //         for (unsigned k = 2; k < 4; ++k) {
    //             coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_C_OFF + k].resize((cols - 1) * (rows - 1));
    //         }
    //         for (size_t i = 0; i < cols - 1; ++i) {
    //             for (size_t j = 0; j < rows - 1; ++j) {
    //                 // Copy from A to A
    //                 coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_A_OFF][j + i * (rows - 1)] = data[dir][tri].a[j + i * (rows - 1)];
    //                 // Copy from B to B
    //                 for (unsigned k = 0; k < 2; ++k) {
    //                     coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_B_OFF + k][j + i * (rows - 1)] = data[dir][tri].b[j + i * (rows - 1)].data()[k];
    //                 }
    //                 // Copy from C to C
    //                 for (unsigned k = 0; k < 4; ++k) {
    //                     coeffs.arr[dir * PMM_DIR_SIZE + tri * PMM_TRI_SIZE + PMM_C_OFF + k][j + i * (rows - 1)] = data[dir][tri].c[j + i * (rows - 1)].data()[k];
    //                 }
    //             }
    //         }
    //     }
    // }

    // Convert PMMGeodesicsData format to linear array as used in the kernel
    {
        unsigned dir = 0;
        for (; dir < 2; ++dir) {
            size_t C_pitch = (cols - 1) * PMM_COEFF_PITCH;
            C[dir].resize(C_pitch * (rows - 1) * 2); // Upwards
            C[dir].resize(C_pitch * (rows - 1) * 2); // Downwards
            for (unsigned tri = 0; tri < 2; ++tri) {
                for (size_t y = 0; y < rows - 1; ++y) {
                    for (size_t x = 0; x < cols - 1; ++x) {
                        std::memcpy(&C[dir][(PMM_A_OFF + 0) + x * PMM_COEFF_PITCH + y * C_pitch * 2 + C_pitch * tri], &data[dir][tri].a[x + y * (cols - 1)],        PMM_A_SIZE * sizeof(Scalar));
                        std::memcpy(&C[dir][(PMM_B_OFF + 0) + x * PMM_COEFF_PITCH + y * C_pitch * 2 + C_pitch * tri],  data[dir][tri].b[x + y * (cols - 1)].data(), PMM_B_SIZE * sizeof(Scalar));
                        std::memcpy(&C[dir][(PMM_C_OFF + 0) + x * PMM_COEFF_PITCH + y * C_pitch * 2 + C_pitch * tri],  data[dir][tri].c[x + y * (cols - 1)].data(), PMM_C_SIZE * sizeof(Scalar));
                    }
                }
            }
        }
        for (; dir < 4; ++dir) {
            size_t C_pitch = (rows - 1) * PMM_COEFF_PITCH;
            C[dir].resize(C_pitch * (cols - 1) * 2); // Rightwards
            C[dir].resize(C_pitch * (cols - 1) * 2); // Leftwards
            for (unsigned tri = 0; tri < 2; ++tri) {
                for (size_t x = 0; x < rows - 1; ++x) {
                    for (size_t y = 0; y < cols - 1; ++y) {
                        std::memcpy(&C[dir][(PMM_A_OFF + 0) + y * PMM_COEFF_PITCH + x * C_pitch * 2 + C_pitch * tri], &data[dir][tri].a[y + x * (rows - 1)],        PMM_A_SIZE * sizeof(Scalar));
                        std::memcpy(&C[dir][(PMM_B_OFF + 0) + y * PMM_COEFF_PITCH + x * C_pitch * 2 + C_pitch * tri],  data[dir][tri].b[y + x * (rows - 1)].data(), PMM_B_SIZE * sizeof(Scalar));
                        std::memcpy(&C[dir][(PMM_C_OFF + 0) + y * PMM_COEFF_PITCH + x * C_pitch * 2 + C_pitch * tri],  data[dir][tri].c[y + x * (rows - 1)].data(), PMM_C_SIZE * sizeof(Scalar));
                    }
                }
            }
        }
    }

    return true;
}