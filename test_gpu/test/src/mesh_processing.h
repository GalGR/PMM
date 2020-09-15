#pragma once

#include <array>
#include <Eigen/Dense>

template <
    typename Scalar,
    typename DerivedV, typename DerivedF,
    typename DerivedV_UV, typename DerivedV_IMG,
    typename DerivedF_IMG, typename DerivedV_UV_IMG
>
void mesh_to_geometry_image(
    const size_t size,
    const Eigen::MatrixBase<DerivedV> &V,
    const Eigen::MatrixBase<DerivedF> &F,
    const Eigen::MatrixBase<DerivedV_UV> &V_uv,
    Eigen::MatrixBase<DerivedV_IMG> &V_img,
    Eigen::MatrixBase<DerivedF_IMG> &F_img,
    Eigen::MatrixBase<DerivedV_UV_IMG> &V_uv_img);

#include "mesh_processing.inl"