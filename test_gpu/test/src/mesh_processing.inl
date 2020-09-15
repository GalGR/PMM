#pragma once

#include "mesh_processing.h"

#include <limits>
#include <vector>
#include <cmath>
#include <iostream>
#include "Barycentric.h"

static double mesh_processing_min_distance_(std::array<std::array<double, 2>, 3> tri, std::array<double, 2> p)
{
    double min = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i)
    {
        std::array<double, 2> v0 = tri[i];
        std::array<double, 2> v1 = tri[(i + 1) % 3];
        double dist = std::abs(
                          (v1[1] - v0[1]) * p[0] - (v1[0] - v0[0]) * p[1] + (v1[0] * v0[1]) - (v1[1] * v0[0])) /
                      std::sqrt(
                          std::pow(v1[1] - v0[1], 2) + std::pow(v1[0] - v0[0], 2));
        if (dist < min)
            min = dist;
    }
    return min;
}

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
    Eigen::MatrixBase<DerivedV_UV_IMG> &V_uv_img
) {
    // Resize the matrices to the size of the image
    V_img.resize(size * size, 3);
    F_img.resize((size - 1) * (size - 1) * 2, 3);
    V_uv_img.resize(size * size, 2);

    // Miminum error in-case the barycentric coordinates fall outside the face
    std::vector<double> min_dists(size * size, std::numeric_limits<double>::infinity());

    // Visit every vertex
    std::vector<bool> visited(size * size, false);

    // Been inside
    std::vector<bool> inside(size * size, false);

    for (int i = 0; i < F.rows(); ++i)
    {
        // Find the face's bounding box
        double x_min = std::numeric_limits<double>::infinity(),
               y_min = std::numeric_limits<double>::infinity(),
               x_max = -std::numeric_limits<double>::infinity(),
               y_max = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < 3; ++j)
        {
            int idx = F(i, j);
            if (V_uv(idx, 0) < x_min)
            {
                x_min = V_uv(idx, 0);
            }
            if (V_uv(idx, 1) < y_min)
            {
                y_min = V_uv(idx, 1);
            }
            if (V_uv(idx, 0) > x_max)
            {
                x_max = V_uv(idx, 0);
            }
            if (V_uv(idx, 1) > y_max)
            {
                y_max = V_uv(idx, 1);
            }
        }
        int x_start = (int)std::max(std::floor(x_min * (size - 1)), 0.0);
        int y_start = (int)std::max(std::floor(y_min * (size - 1)), 0.0);
        int x_end = (int)std::min(std::ceil(x_max * (size - 1)), (double)(size - 1));
        int y_end = (int)std::min(std::ceil(y_max * (size - 1)), (double)(size - 1));
        for (int y = y_start; y <= y_end; ++y)
        {
            for (int x = x_start; x <= x_end; ++x)
            {
                double u = ((double)x) / (size - 1);
                double v = ((double)y) / (size - 1);
                Bary bary(
                    std::array<double, 2>{u, v},
                    std::array<std::array<double, 2>, 3>{
                        std::array<double, 2>{V_uv(F(i, 0), 0), V_uv(F(i, 0), 1)},
                        std::array<double, 2>{V_uv(F(i, 1), 0), V_uv(F(i, 1), 1)},
                        std::array<double, 2>{V_uv(F(i, 2), 0), V_uv(F(i, 2), 1)}});
                double dist = mesh_processing_min_distance_(
                    std::array<std::array<double, 2>, 3>{
                        std::array<double, 2>{V_uv(F(i, 0), 0), V_uv(F(i, 0), 1)},
                        std::array<double, 2>{V_uv(F(i, 1), 0), V_uv(F(i, 1), 1)},
                        std::array<double, 2>{V_uv(F(i, 2), 0), V_uv(F(i, 2), 1)}},
                    std::array<double, 2>{u, v});
                int idx = x + y * size;
                bool bary_inside = bary.isInside();
                if (!inside[idx] && (bary_inside || dist < min_dists[idx]))
                {
                    if (dist < min_dists[idx])
                        min_dists[idx] = dist;
                    if (bary_inside)
                        inside[idx] = true;
                    std::array<double, 3> xyz = bary(
                        std::array<std::array<double, 3>, 3>{
                            std::array<double, 3>{V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2)},
                            std::array<double, 3>{V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2)},
                            std::array<double, 3>{V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2)}});
                    V_img.row(idx) << ((Scalar)xyz[0]), ((Scalar)xyz[1]), ((Scalar)xyz[2]);
                    V_uv_img.row(idx) << ((Scalar)u), ((Scalar)v);
                    visited[idx] = true;
                }
            }
        }
    }
    // Check if visited every vertex
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            if (!visited[x + y * size])
            {
                std::cout << "Not visited image vertex: x=" << x << ", y=" << y << std::endl;
            }
        }
    }
    // Check if been inside every vertex
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            if (!inside[x + y * size])
            {
                std::cout << "Not inside image vertex: x=" << x << ", y=" << y << std::endl;
            }
        }
    }
    // Define the faces
    int idx = 0;
    for (int y = 0; y < size - 1; ++y)
    {
        for (int x = 0; x < size - 1; ++x)
        {
            F_img.row(idx++) << x + y * size, (x + 1) + y * size, (x + 1) + (y + 1) * size;
            F_img.row(idx++) << x + y * size, (x + 1) + (y + 1) * size, x + (y + 1) * size;
        }
    }
}