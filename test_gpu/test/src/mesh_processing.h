#pragma once

#include <array>
#include <Eigen/Dense>

void mesh_to_geometry_image(
    const size_t size,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &V_uv,
    Eigen::MatrixXd &V_img,
    Eigen::MatrixXi &F_img,
    Eigen::MatrixXd &V_uv_img);