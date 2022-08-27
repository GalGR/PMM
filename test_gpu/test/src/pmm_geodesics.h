#ifndef PMM_GEODESICS_H_
#define PMM_GEODESICS_H_

#include "pmm.h"

#include <vector>
#include <array>
#include <stddef.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "pmm_geodesics_constants.h"

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

        inline void resize(size_t rows, size_t cols) {
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

            inline void resize(size_t s) {
                a.resize(s);
                b.resize(s);
                c.resize(s);
            }
        } left, right;

        inline void resize(size_t lines, size_t width) {
            left.resize(lines * (width - 1));
            right.resize(lines * (width - 1));
        }

        inline const DataTriangle &operator [](int i) const {
            return (&left)[i];
        }
        inline DataTriangle &operator [](int i) {
            return (&left)[i];
        }
    } upwards, downwards, rightwards, leftwards;

    inline void resize(size_t height, size_t width) {
        this->rows = height;
        this->cols = width;
        V_row.resize(height, width);
        V_col.resize(height, width);
        upwards.resize(height - 1, width);
        downwards.resize(height - 1, width);
        rightwards.resize(width - 1, height);
        leftwards.resize(width - 1, height);
    }

    inline const DataDirection &operator [](int i) const {
        return (&upwards)[i];
    }
    inline DataDirection &operator [](int i) {
        return (&upwards)[i];
    }
};

template <typename Scalar, typename DerivedV>
bool pmm_geodesics_precompute(
    PMMGeodesicsData<Scalar> &data,
    size_t rows, size_t cols,
    const Eigen::MatrixBase<DerivedV> &V,
    std::array<std::vector<Scalar>, 4> &C,
    bool ignore_non_acute_triangles = false);

template <typename Scalar, typename DerivedV>
bool pmm_geodesics_precompute(
    size_t rows, size_t cols,
    const Eigen::MatrixBase<DerivedV> &V,
    std::array<std::vector<Scalar>, 4> &C,
    bool ignore_non_acute_triangles = false);

template <typename Scalar>
void pmm_geodesics_solve(
    size_t rows, size_t cols,
    int maxGridWidth,
    int maxThreads,
    int warpSize,
    size_t maxSharedMem,
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
    size_t omega);

#include "pmm_geodesics_precompute.inl"

#endif