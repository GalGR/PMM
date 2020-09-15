#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/PI.h>
#include <igl/read_triangle_mesh.h>

#include "data_shared_path.h"

#include <limits>
#include <cmath>
#include <array>
#include "Barycentric.h"

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd V_uv;
Eigen::MatrixXd V_uv_img;
Eigen::MatrixXd V_img;
Eigen::MatrixXi F_img;

unsigned long img_size;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  bool reload_mesh = (key > '0' && key <= '9');
  if (reload_mesh) {
    bool show_lines = viewer.data().show_lines;
    bool show_texture = viewer.data().show_texture;
    viewer.data().clear();
    viewer.data().show_lines = show_lines;
    viewer.data().show_texture = show_texture;
  }
  if (key == '1')
  {
    // Plot the 3D mesh
    viewer.data().set_mesh(V,F);
    viewer.data().set_uv(V_uv);
    viewer.core().align_camera_center(V,F);
  }
  else if (key == '2')
  {
    // Plot the mesh in 2D using the UV coordinates as vertex coordinates
    viewer.data().set_mesh(V_uv,F);
    viewer.data().set_uv(V_uv);
    viewer.core().align_camera_center(V_uv,F);
  }
  else if (key == '3')
  {
    // Plot the geometry image
    viewer.data().set_mesh(V_img, F_img);
    viewer.data().set_uv(V_uv_img);
    viewer.core().align_camera_center(V_img, F_img);
  }
  else if (key == '4')
  {
    // Plot the geometry image
    viewer.data().set_mesh(V_uv_img, F_img);
    viewer.data().set_uv(V_uv_img);
    viewer.core().align_camera_center(V_uv_img, F_img);
  }
  else if (key == '`')
  {
    viewer.data().show_texture = !(viewer.data().show_texture);
  }

  if (reload_mesh) viewer.data().compute_normals();

  return false;
}

double min_distance(std::array<std::array<double, 2>, 3> tri, std::array<double, 2> p) {
  double min = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 3; ++i) {
    std::array<double, 2> v0 = tri[i];
    std::array<double, 2> v1 = tri[(i + 1) % 3];
    double dist = std::abs(
      (v1[1] - v0[1]) * p[0] - (v1[0] - v0[0]) * p[1]
      + (v1[0] * v0[1]) - (v1[1] * v0[0])
    ) / std::sqrt(
      std::pow(v1[1] - v0[1], 2) + std::pow(v1[0] - v0[0], 2)
    );
    if (dist < min) min = dist;
  }
  return min;
}

void mesh_to_geometry_image(
  const unsigned long size,
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &V_uv,
  Eigen::MatrixXd &V_img,
  Eigen::MatrixXi &F_img,
  Eigen::MatrixXd &V_uv_img)
{
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

  for (int i = 0; i < F.rows(); ++i) {
    // Find the face's bounding box
    double x_min = std::numeric_limits<double>::infinity(),
      y_min = std::numeric_limits<double>::infinity(),
      x_max = -std::numeric_limits<double>::infinity(),
      y_max = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < 3; ++j) {
      int idx = F(i, j);
      if (V_uv(idx, 0) < x_min) {
        x_min = V_uv(idx, 0);
      }
      if (V_uv(idx, 1) < y_min) {
        y_min = V_uv(idx, 1);
      }
      if (V_uv(idx, 0) > x_max) {
        x_max = V_uv(idx, 0);
      }
      if (V_uv(idx, 1) > y_max) {
        y_max = V_uv(idx, 1);
      }
    }
    int x_start = (int)std::max(std::floor(x_min * (size - 1)), 0.0);
    int y_start = (int)std::max(std::floor(y_min * (size - 1)), 0.0);
    int x_end = (int)std::min(std::ceil(x_max * (size - 1)), (double)(size - 1));
    int y_end = (int)std::min(std::ceil(y_max * (size - 1)), (double)(size - 1));
    for (int y = y_start; y <= y_end; ++y) {
      for (int x = x_start; x <= x_end; ++x) {
        double u = ((double)x) / (size - 1);
        double v = ((double)y) / (size - 1);
        Bary bary(
          std::array<double, 2>{u, v},
          std::array<std::array<double, 2>, 3>{
            std::array<double, 2>{V_uv(F(i, 0), 0), V_uv(F(i, 0), 1)},
            std::array<double, 2>{V_uv(F(i, 1), 0), V_uv(F(i, 1), 1)},
            std::array<double, 2>{V_uv(F(i, 2), 0), V_uv(F(i, 2), 1)}
        });
        double dist = min_distance(
          std::array<std::array<double, 2>, 3>{
            std::array<double, 2>{V_uv(F(i, 0), 0), V_uv(F(i, 0), 1)},
            std::array<double, 2>{V_uv(F(i, 1), 0), V_uv(F(i, 1), 1)},
            std::array<double, 2>{V_uv(F(i, 2), 0), V_uv(F(i, 2), 1)}
          },
          std::array<double, 2>{u, v}
        );
        int idx = x + y * size;
        bool bary_inside = bary.isInside();
        if (!inside[idx] && (bary_inside || dist < min_dists[idx])) {
          if (dist < min_dists[idx]) min_dists[idx] = dist;
          if (bary_inside) inside[idx] = true;
          std::array<double, 3> xyz = bary(
            std::array<std::array<double, 3>, 3>{
              std::array<double, 3>{V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2)},
              std::array<double, 3>{V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2)},
              std::array<double, 3>{V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2)}
            }
          );
          V_img.row(idx) << xyz[0], xyz[1], xyz[2];
          V_uv_img.row(idx) << u, v;
          visited[idx] = true;
        }
      }
    }
  }
  // Check if visited every vertex
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      if (!visited[x + y * size]) {
        std::cout << "Not visited image vertex: x=" << x << ", y=" << y << std::endl;
      }
    }
  }
  // Check if been inside every vertex
  // Check if visited every vertex
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      if (!inside[x + y * size]) {
        std::cout << "Not inside image vertex: x=" << x << ", y=" << y << std::endl;
      }
    }
  }
  // Define the faces
  int idx = 0;
  for (int y = 0; y < size - 1; ++y) {
    for (int x = 0; x < size - 1; ++x) {
      F_img.row(idx++) << x + y * size, (x + 1) + y * size, (x + 1) + (y + 1) * size;
      F_img.row(idx++) << x + y * size, (x + 1) + (y + 1) * size, x + (y + 1) * size;
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc < 4) {
    std::cerr << "Not enough arguments" << std::endl;
    exit(1);
  }

  // Geometry image size
  img_size = atoi(argv[2]);

  // Load a mesh in the correct format
  if (!igl::read_triangle_mesh(std::string(DATA_SHARED_PATH "/") + std::string(argv[1]), V, F)) {
    std::cerr << "Couldn't load object" << std::endl;
    exit(1);
  }

  // Find the open boundary
  Eigen::VectorXi bnd;
  try { igl::boundary_loop(F,bnd); }
  catch (...) { std::cerr << "Boundary loop crashed" << std::endl; }
  if (bnd.size() == 0) {
    std::cerr << "The model has no boundary" << std::endl;
    exit(1);
  }

  // Map the boundary to a square, preserving edge proportions
  Eigen::MatrixXd bnd_uv;
  // igl::map_vertices_to_circle(V, bnd, bnd_uv);
  auto map_vertices_to_square = [&] (const Eigen::MatrixXd &V, const Eigen::VectorXi &bnd, Eigen::MatrixXd& UV) {
    // Get sorted list of boundary vertices
    std::vector<int> interior,map_ij;
    map_ij.resize(V.rows());

    std::vector<bool> isOnBnd(V.rows(),false);
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
    std::vector<double> len(bnd.size() + 1);
    len[0] = 0.;

    for (int i = 1; i < bnd.size(); i++)
    {
      len[i] = len[i-1] + (V.row(bnd[i-1]) - V.row(bnd[i])).norm();
    }
    double total_len = len[bnd.size()-1] + (V.row(bnd[0]) - V.row(bnd[bnd.size()-1])).norm();
    len[bnd.size()] = total_len;

    // Find the 4 corners
    Eigen::VectorXi corners(5);
    double edge_len = total_len / 4;
    double corner_len = 0.0;

    for (int corner = 0, i = 0; corner < 4 && i < bnd.size(); ++i) {
      if (len[i] >= corner_len) {
        corners(corner) = i;
        ++corner;
        corner_len += edge_len;
      }
    }
    corners(4) = bnd.size();

    UV.resize(bnd.size(),2);
    for (int corner = 0, i = 0; corner < 4 && i < bnd.size(); ++i) {
      int next_corner = corner + 1;
      double frac = (len[i] - len[corners(corner)]) / (len[corners(next_corner)] - len[corners(corner)]);
      switch (corner) {
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
      if (i == corners(next_corner) - 1) {
        ++corner;
      }
    }
  };
  map_vertices_to_square(V, bnd, bnd_uv);

  // Harmonic parametrization for the internal vertices
  igl::harmonic(V,F,bnd,bnd_uv,atoi(argv[3]),V_uv);

  mesh_to_geometry_image(img_size, V, F, V_uv, V_img, F_img, V_uv_img);

  // Scale UV to make the texture more clear
  V_uv *= 5;
  V_uv_img *= 5;

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_uv(V_uv);
  viewer.callback_key_down = &key_down;

  // Enable wireframe
  viewer.data().show_lines = true;

  // Draw checkerboard texture
  viewer.data().show_texture = true;

  // Launch the viewer
  viewer.launch();
}
