#pragma once

#include "scalar_types.h"

#include <Eigen/Dense>

void map_vertices_to_square(const Eigen::MatrixXf &V, const Eigen::VectorXi &bnd, Eigen::MatrixXf &UV);