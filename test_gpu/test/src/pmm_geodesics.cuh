#ifndef PMM_GEODESICS_H_
#define PMM_GEODESICS_H_

#include "pmm.h"

#include <vector>
#include <array>
#include <stddef.h>
#include <Eigen/Dense>

// 56 Arrays = 4 Directions * 2 Triangles * (1 Float (a) + 2 Floats (b) + 4 Floats (c))
// Example: Height = 1024, Width = 1024, Scalar = float
//    234,881,024 Bytes = 56 Arrays * 1024 Height * 1024 Width * 4 Bytes (float)
#define PMM_ARRAYS 56
enum PMM_OFFS {
    PMM_UPWARDS_LEFT_A = 0,
    PMM_UPWARDS_LEFT_B_0,
    PMM_UPWARDS_LEFT_B_1,
    PMM_UPWARDS_LEFT_C_0,
    PMM_UPWARDS_LEFT_C_1,
    PMM_UPWARDS_LEFT_C_2,
    PMM_UPWARDS_LEFT_C_3,
    PMM_UPWARDS_RIGHT_A,
    PMM_UPWARDS_RIGHT_B_0,
    PMM_UPWARDS_RIGHT_B_1,
    PMM_UPWARDS_RIGHT_C_0,
    PMM_UPWARDS_RIGHT_C_1,
    PMM_UPWARDS_RIGHT_C_2,
    PMM_UPWARDS_RIGHT_C_3,
    PMM_DOWNWARDS_LEFT_A,
    PMM_DOWNWARDS_LEFT_B_0,
    PMM_DOWNWARDS_LEFT_B_1,
    PMM_DOWNWARDS_LEFT_C_0,
    PMM_DOWNWARDS_LEFT_C_1,
    PMM_DOWNWARDS_LEFT_C_2,
    PMM_DOWNWARDS_LEFT_C_3,
    PMM_DOWNWARDS_RIGHT_A,
    PMM_DOWNWARDS_RIGHT_B_0,
    PMM_DOWNWARDS_RIGHT_B_1,
    PMM_DOWNWARDS_RIGHT_C_0,
    PMM_DOWNWARDS_RIGHT_C_1,
    PMM_DOWNWARDS_RIGHT_C_2,
    PMM_DOWNWARDS_RIGHT_C_3,
    PMM_RIGHTWARDS_LEFT_A,
    PMM_RIGHTWARDS_LEFT_B_0,
    PMM_RIGHTWARDS_LEFT_B_1,
    PMM_RIGHTWARDS_LEFT_C_0,
    PMM_RIGHTWARDS_LEFT_C_1,
    PMM_RIGHTWARDS_LEFT_C_2,
    PMM_RIGHTWARDS_LEFT_C_3,
    PMM_RIGHTWARDS_RIGHT_A,
    PMM_RIGHTWARDS_RIGHT_B_0,
    PMM_RIGHTWARDS_RIGHT_B_1,
    PMM_RIGHTWARDS_RIGHT_C_0,
    PMM_RIGHTWARDS_RIGHT_C_1,
    PMM_RIGHTWARDS_RIGHT_C_2,
    PMM_RIGHTWARDS_RIGHT_C_3,
    PMM_LEFTWARDS_LEFT_A,
    PMM_LEFTWARDS_LEFT_B_0,
    PMM_LEFTWARDS_LEFT_B_1,
    PMM_LEFTWARDS_LEFT_C_0,
    PMM_LEFTWARDS_LEFT_C_1,
    PMM_LEFTWARDS_LEFT_C_2,
    PMM_LEFTWARDS_LEFT_C_3,
    PMM_LEFTWARDS_RIGHT_A,
    PMM_LEFTWARDS_RIGHT_B_0,
    PMM_LEFTWARDS_RIGHT_B_1,
    PMM_LEFTWARDS_RIGHT_C_0,
    PMM_LEFTWARDS_RIGHT_C_1,
    PMM_LEFTWARDS_RIGHT_C_2,
    PMM_LEFTWARDS_RIGHT_C_3,
};
// 14 Direction Size = 2 Triangles * (1 Float (a) + 2 Floats (b) + 4 Floats (c))
#define PMM_DIR_SIZE 14
// 7 Triangle Size = 1 Float (a) + 2 Floats (b) + 4 Floats (c)
#define PMM_TRI_SIZE 7
// No Skip (first in the triangle)
#define PMM_A_OFF 0
// 1 Skip = 1 Float (a)
#define PMM_B_OFF 1
// 3 Skip = 1 Float (a) + 2 Floats (b)
#define PMM_C_OFF 3

// Coefficients array elements pitch
#define PMM_COEFF_PITCH 8

template <typename Scalar>
struct PMMCoeffs {
    std::array<std::vector<Scalar>, PMM_ARRAYS> arr;
};

template <typename Scalar>
struct PMMGeodesicsData {
    size_t rows;
    size_t cols;
    template <decltype(Eigen::RowMajor) Major>
    struct Vertices {
        Eigen::Matrix<Scalar, -1, -1, Major> X;
        Eigen::Matrix<Scalar, -1, -1, Major> Y;
        Eigen::Matrix<Scalar, -1, -1, Major> Z;

        PMM_INLINE void resize(size_t rows, size_t cols) {
            X.resize(rows, cols);
            Y.resize(rows, cols);
            Z.resize(rows, cols);
        }
    };
    Vertices<Eigen::RowMajor> V_row;
    Vertices<Eigen::ColMajor> V_col;
    struct DataDirection {
        struct DataTriangle {
            // A -- x^2 coefficient (1^TQ1)
            std::vector<Scalar> a;
            // B -- x^1 coefficient (1^TQt -- without t)
            std::vector<Eigen::Matrix<Scalar, 1, 2> > b;
            // C -- x^0 coefficient (t^TQt -- without t)
            std::vector<Eigen::Matrix<Scalar, 2, 2> > c; // c == Q == E^-1
            // E == X^TX == ( e_11  e_12 )
            //              ( e_21  e_22 )
            // X   == ( x_1  x_2 ) == ( x_1_x  x_2_x )
            //                        ( x_1_y  x_2_y )
            //                        ( x_1_z  x_2_z )
            // X^T == ( x_1^T )    == ( x_1_x  x_1_y  x_1_z )
            //        ( x_2^T )       ( x_2_x  x_2_y  x_2_z )
            // <x_1x_0x_2 acute:   e_12 > 0
            // <x_1x_0x_2 obtuse:  e_12 < 0

            PMM_INLINE void resize(size_t s) {
                a.resize(s);
                b.resize(s);
                c.resize(s);
            }
        } left, right;

        PMM_INLINE void resize(size_t lines, size_t width) {
            left.resize(lines * (width - 1));
            right.resize(lines * (width - 1));
        }

        PMM_INLINE const DataTriangle &operator [](int i) const {
            return (&left)[i];
        }
        PMM_INLINE DataTriangle &operator [](int i) {
            return (&left)[i];
        }
    } upwards, downwards, rightwards, leftwards;

    PMM_INLINE void resize(size_t height, size_t width) {
        this->rows = height;
        this->cols = width;
        V_row.resize(height, width);
        V_col.resize(height, width);
        upwards.resize(height - 1, width);
        downwards.resize(height - 1, width);
        rightwards.resize(width - 1, height);
        leftwards.resize(width - 1, height);
    }

    PMM_INLINE const DataDirection &operator [](int i) const {
        return (&upwards)[i];
    }
    PMM_INLINE DataDirection &operator [](int i) {
        return (&upwards)[i];
    }
};

template < typename Scalar, typename DerivedV, typename DerivedF >
PMM_INLINE bool pmm_geodesics_precompute(
    size_t rows, size_t cols,
    const Eigen::MatrixBase<DerivedV> & V,
    const Eigen::MatrixBase<DerivedF> & F,
    PMMCoeffs &coeffs,
    bool ignore_non_acute_triangles = false);

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
    bool D_empty = true);

#include "pmm_geodesics_precompute.inl"

#include "pmm_geodesics_solve.inl"

#endif