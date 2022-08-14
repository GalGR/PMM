#include "main_funcs.h"

#include <vector>

void map_vertices_to_square(const Eigen::MatrixXf &V, const Eigen::VectorXi &bnd, Eigen::MatrixXf &UV)
{
    // Get sorted list of boundary vertices
    std::vector<int> interior, map_ij;
    map_ij.resize(V.rows());

    std::vector<bool> isOnBnd(V.rows(), false);
    for (int i = 0; i < bnd.size(); i++)
    {
        isOnBnd[bnd[i]] = true;
        map_ij[bnd[i]] = i;
    }

    for (int i = 0; i < (int)isOnBnd.size(); i++)
    {
        if (!isOnBnd[i])
        {
            map_ij[i] = interior.size();
            interior.push_back(i);
        }
    }

    // Map boundary to unit square
    std::vector<Scalar> len(bnd.size() + 1);
    len[0] = 0.;

    for (int i = 1; i < bnd.size(); i++)
    {
        len[i] = len[i - 1] + (V.row(bnd[i - 1]) - V.row(bnd[i])).norm();
    }
    Scalar total_len = len[bnd.size() - 1] + (V.row(bnd[0]) - V.row(bnd[bnd.size() - 1])).norm();
    len[bnd.size()] = total_len;

    // Find the 4 corners
    Eigen::VectorXi corners(5);
    Scalar edge_len = total_len / 4;
    Scalar corner_len = 0.0;

    for (int corner = 0, i = 0; corner < 4 && i < bnd.size(); ++i)
    {
        if (len[i] >= corner_len)
        {
            corners(corner) = i;
            ++corner;
            corner_len += edge_len;
        }
    }
    corners(4) = bnd.size();

    UV.resize(bnd.size(), 2);
    for (int corner = 0, i = 0; corner < 4 && i < bnd.size(); ++i)
    {
        int next_corner = corner + 1;
        Scalar frac = (len[i] - len[corners(corner)]) / (len[corners(next_corner)] - len[corners(corner)]);
        switch (corner)
        {
        case 0:
            UV.row(map_ij[bnd[i]]) << frac, 0.0;
            break;
        case 1:
            UV.row(map_ij[bnd[i]]) << 1.0, frac;
            break;
        case 2:
            UV.row(map_ij[bnd[i]]) << (1.0 - frac), 1.0;
            break;
        case 3:
            UV.row(map_ij[bnd[i]]) << 0.0, (1.0 - frac);
            break;
        }
        if (i == corners(next_corner) - 1)
        {
            ++corner;
        }
    }
}