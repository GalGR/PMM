#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/triangulated_grid.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/avg_edge_length.h>
#include <igl/isolines_map.h>
#include <igl/opengl/create_shader_program.h>
#include <igl/opengl/destroy_shader_program.h>
#include <igl/PI.h>

#include "pmm.h"

#include "data_shared_path.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>
#include <array>
#include "mesh_processing.h"
#include "plf_nanotimer/plf_nanotimer.h"
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#ifdef MATLAB_DEBUG
  #include "MatlabUtils/MatlabInterface.h"
  #include "MatlabUtils/GMM_Macros.h"
  #include "MatlabUtils/MatlabGMMDataExchange.h"
#endif

#ifdef MATRIX_FILE
bool is_matrix_file = false;
std::string matrix_filename;
#endif

#define TIMER_START(MSG) do { \
                             std::cout << MSG "..." << std::endl; timer.start(); \
                         } while(0)
#define TIMER_END()      do { \
                             std::cout << "Done!\t" << timer.get_elapsed_s() << "s" << std::endl; \
                         } while(0)
#define TIMER_ERROR(MSG) do { \
                             std::cerr << "Error\t" << timer.get_elapsed_s() << "s" << ": " << MSG << std::endl; \
                         } while(0)

#ifdef MATRIX_FILE
  template <typename Data_T>
  void bin_write(std::ofstream &file, const Data_T &data) {
    file.write((const char*)(&data), sizeof(data));
    file.flush();
  }
  template <typename Data_T>
  void bin_write_arr(std::ofstream &file, const Data_T *ptr, size_t len) {
    file.write((const char*)ptr, sizeof(ptr[0]) * len);
    file.flush();
  }
#endif

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd V_uv;
Eigen::MatrixXd V_uv_img;
Eigen::MatrixXd V_img;
Eigen::MatrixXi F_img;

std::string model_name;
size_t rows;
size_t cols;
size_t img_len;
int harmonic_const;
size_t N_iters; // Number of PMM iterations

bool start_with_source = false;
std::vector<size_t> start_source;

bool ignore_non_acute_triangles = false;

plf::nanotimer timer;

void set_colormap(igl::opengl::glfw::Viewer & viewer)
{
  // Original code generated bright-red to dark-red colormap
  // const int num_intervals = 30;
  // Eigen::MatrixXd CM(num_intervals,3);
  // // Colormap texture
  // for(int i = 0;i<num_intervals;i++)
  // {
  //   double t = double(num_intervals - i - 1)/double(num_intervals-1);
  //   CM(i,0) = std::max(std::min(2.0*t-0.0,1.0),0.0);
  //   CM(i,1) = std::max(std::min(2.0*t-1.0,1.0),0.0);
  //   CM(i,2) = std::max(std::min(6.0*t-5.0,1.0),0.0);
  // }
  // Switched to Matlab's Jet colormap
  const int num_intervals = 30;
  Eigen::MatrixXd CM(num_intervals,3);
  igl::colormap(igl::COLOR_MAP_TYPE_JET,Eigen::VectorXd::LinSpaced(num_intervals,0,1).eval(),1,0,CM);
  igl::isolines_map(Eigen::MatrixXd(CM),CM);
  viewer.data().set_colormap(CM);
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  static plf::nanotimer timer;
  bool reload_mesh = (key > '0' && key <= '9');
  if (reload_mesh) {
    TIMER_START("Reloading mesh");
    bool show_lines = viewer.data().show_lines;
    bool show_texture = viewer.data().show_texture;
    viewer.data().clear();
    viewer.data().show_lines = show_lines;
    viewer.data().show_texture = show_texture;
  }
  switch (key) {
  case '1':
    // Plot the 3D mesh
    viewer.data().set_mesh(V,F);
    viewer.data().set_uv(V_uv);
    viewer.core().align_camera_center(V,F);
    break;
  case '2':
    // Plot the mesh in 2D using the UV coordinates as vertex coordinates
    viewer.data().set_mesh(V_uv,F);
    viewer.data().set_uv(V_uv);
    viewer.core().align_camera_center(V_uv,F);
    break;
  case '3':
    // Plot the geometry image
    viewer.data().set_mesh(V_img, F_img);
    viewer.data().set_uv(V_uv_img);
    viewer.core().align_camera_center(V_img, F_img);
    break;
  case '4':
    // Plot the geometry image's UV
    viewer.data().set_mesh(V_uv_img, F_img);
    viewer.data().set_uv(V_uv_img);
    viewer.core().align_camera_center(V_uv_img, F_img);
    break;
  case '`':
    // Toggle texture visibility
    viewer.data().show_texture = !(viewer.data().show_texture);
    break;
  }

  if (reload_mesh) {
    viewer.data().compute_normals();
    TIMER_END();
  }

  return false;
}

int main(int argc, char *argv[])
{
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("model,m", po::value<std::string>(&model_name), "model")
    ("height,h", po::value<size_t>(&rows), "geometric image height/rows")
    ("width,w", po::value<size_t>(&cols), "geometric image width/columns")
    ("harmonic,H", po::value<int>(&harmonic_const)->default_value(1), "harmonic parameterization constant")
    ("iterations,i", po::value<size_t>(&N_iters)->default_value(1), "number of PMM iterations")
    ("source,s", po::value<std::vector<size_t> >(&start_source), "start source vertices")
    ("ignore-non-acute", "the precomputation ignores non acute triangles")
    #ifdef MATRIX_FILE
      ("file,f", po::value<std::string>(&matrix_filename), "file to write the matrix in")
    #endif
  ;
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("ignore-non-acute")) {
    ignore_non_acute_triangles = true;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
  }

  if (!vm.count("model") && !vm.count("height") && !vm.count("width")) {
    std::cerr << "'model', 'height' and 'width' are required arguments" << std::endl;
    std::cout << desc << std::endl;
    exit(EXIT_FAILURE);
  }

  // Geometry image size
  if (rows != cols) {
    std::cerr << "different rows and cols are not yet supported" << std::endl;
    exit(EXIT_FAILURE);
  }
  img_len = rows * cols;

  // Check for start source
  if (vm.count("source")) {
    start_with_source = true;
  }

  #ifdef MATRIX_FILE
    // Check for matrix file
    if (vm.count("file")) {
      is_matrix_file = true;
    }
    std::ofstream matrix_file;
    std::streampos matrix_file_rows_cols_pos;
    std::streampos matrix_file_V_pos;
    std::streampos matrix_file_coeff_pos;
    std::streampos matrix_file_D_pos;
    if (is_matrix_file) {
      try {
        matrix_file.open(matrix_filename, std::ios::out | std::ios::binary | std::ios::trunc);
      } catch (const std::exception &e) {
        std::cerr << "Error: Couldn't open matrix file for writing" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  #endif

  std::cout << std::endl;

  // Load a mesh
  std::cout << "Loading " << model_name << " triangle mesh..." << std::endl;
  timer.start();
  if (!igl::read_triangle_mesh(model_name, V, F)) {
    TIMER_ERROR("Couldn't load object");
    exit(EXIT_FAILURE);
  }
  TIMER_END();

  // Find the open boundary
  TIMER_START("Finding open boundary");
  Eigen::VectorXi bnd;
  try { igl::boundary_loop(F,bnd); }
  catch (...) { TIMER_ERROR("Boundary loop crashed"); }
  if (bnd.size() == 0) {
    TIMER_ERROR("The model has no boundary");
    exit(EXIT_FAILURE);
  }
  TIMER_END();

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
  TIMER_START("Map boundary to square");
  map_vertices_to_square(V, bnd, bnd_uv);
  TIMER_END();

  // Harmonic parametrization for the internal vertices
  TIMER_START("Generating harmonic parameterization");
  igl::harmonic(V,F,bnd,bnd_uv,harmonic_const,V_uv);
  TIMER_END();

  TIMER_START("Generating geometry image");
  mesh_to_geometry_image(rows, V, F, V_uv, V_img, F_img, V_uv_img);
  TIMER_END();

  #ifdef MATRIX_FILE
    if (is_matrix_file) {
      matrix_file_rows_cols_pos = matrix_file.tellp();
      std::cout << "Writing to \"" << matrix_filename << "\" the number of rows (" << rows << ")" << std::endl;
      try {
        bin_write(matrix_file, rows);
      } catch (const std::exception &e) {
        std::cerr << "Error: Write to matrix file failed" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      std::cout << "Written " << sizeof(rows) << " bytes (seek=" << matrix_file.tellp() << ")" << std::endl;
      std::cout << "Writing to \"" << matrix_filename << "\" the number of cols (" << cols << ")" << std::endl;
      try {
        bin_write(matrix_file, cols);
      } catch (const std::exception &e) {
        std::cerr << "Error: Write to matrix file failed" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      std::cout << "Written " << sizeof(cols) << " bytes (seek=" << matrix_file.tellp() << ")" << std::endl;
      matrix_file_V_pos = matrix_file.tellp();
      TIMER_START("Writing V matrix");
      try {
        bin_write_arr(matrix_file, V_img.data(), img_len * 3);
      } catch (const std::exception &e) {
        TIMER_ERROR("Write to matrix file failed");
        std::cerr << "Exception: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      std::cout << "Written " << img_len * 3 * sizeof(V_img.data()[0]) << " bytes (seek=" << matrix_file.tellp() << ")" << std::endl;
      TIMER_END();
      matrix_file_coeff_pos = matrix_file.tellp();
    }
  #endif

  // Scale UV to make the texture more clear
  TIMER_START("Scaling UV coordinates");
  V_uv *= 5;
  V_uv_img *= 5;
  TIMER_END();

  // Precomputation
  PMMGeodesicsData<double> data;
  double t = std::pow(igl::avg_edge_length(V_img,F_img),2);
  const auto precompute = [&]()
  {
    if(!pmm_geodesics_precompute(rows, cols, V_img,F_img,data, ignore_non_acute_triangles))
    {
      TIMER_ERROR("pmm_geodesics_precompute failed");
      exit(EXIT_FAILURE);
    };
  };
  TIMER_START("Precomputing geodesics data");
  precompute();
  TIMER_END();

  #ifdef MATRIX_FILE
    if (is_matrix_file) {
      #define STRINGIZE(X) #X
      #define MATRIX_FILE_COEFF_MACRO(ABC, DIRECTION, L_R) do { \
          TIMER_START("Writing '" STRINGIZE(ABC) "' " STRINGIZE(DIRECTION) " " STRINGIZE(L_R) " coefficients matrix"); \
          const auto &dat = data.DIRECTION.L_R.ABC; \
          try { \
              bin_write_arr(matrix_file, dat.data(), dat.size()); \
          } catch (const std::exception &e) { \
              TIMER_ERROR("Write to matrix file failed"); \
              std::cerr << "Exception: " << e.what() << std::endl; \
              exit(EXIT_FAILURE); \
          } \
          std::cout << "Written " << dat.size() * sizeof(dat.data()[0]) << " bytes (seek=" << matrix_file.tellp() << ")" << std::endl; \
          TIMER_END(); \
      } while(0)
      MATRIX_FILE_COEFF_MACRO(a, upwards, left);
      MATRIX_FILE_COEFF_MACRO(b, upwards, left);
      MATRIX_FILE_COEFF_MACRO(c, upwards, left);
      MATRIX_FILE_COEFF_MACRO(a, upwards, right);
      MATRIX_FILE_COEFF_MACRO(b, upwards, right);
      MATRIX_FILE_COEFF_MACRO(c, upwards, right);
      MATRIX_FILE_COEFF_MACRO(a, downwards, left);
      MATRIX_FILE_COEFF_MACRO(b, downwards, left);
      MATRIX_FILE_COEFF_MACRO(c, downwards, left);
      MATRIX_FILE_COEFF_MACRO(a, downwards, right);
      MATRIX_FILE_COEFF_MACRO(b, downwards, right);
      MATRIX_FILE_COEFF_MACRO(c, downwards, right);
      MATRIX_FILE_COEFF_MACRO(a, rightwards, left);
      MATRIX_FILE_COEFF_MACRO(b, rightwards, left);
      MATRIX_FILE_COEFF_MACRO(c, rightwards, left);
      MATRIX_FILE_COEFF_MACRO(a, rightwards, right);
      MATRIX_FILE_COEFF_MACRO(b, rightwards, right);
      MATRIX_FILE_COEFF_MACRO(c, rightwards, right);
      MATRIX_FILE_COEFF_MACRO(a, leftwards, left);
      MATRIX_FILE_COEFF_MACRO(b, leftwards, left);
      MATRIX_FILE_COEFF_MACRO(c, leftwards, left);
      MATRIX_FILE_COEFF_MACRO(a, leftwards, right);
      MATRIX_FILE_COEFF_MACRO(b, leftwards, right);
      MATRIX_FILE_COEFF_MACRO(c, leftwards, right);
      #undef STRINGIZE
      #undef MATRIX_FILE_COEFF_MACRO

      matrix_file_D_pos = matrix_file.tellp();
    }
  #endif

  igl::opengl::glfw::Viewer viewer;
  bool down_on_mesh = false;
  const auto update = [&]()->bool
  {
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
      viewer.core().proj, viewer.core().viewport, V_img, F_img, fid, bc))
    {
      TIMER_START("Running PMM");
      Eigen::VectorXd D;
      // if big mesh, just use closest vertex. Otherwise, blend distances to
      // vertices of face using barycentric coordinates.
      // if(F_img.rows()>100000)
      {
        // 3d position of hit
        const Eigen::RowVector3d m3 =
          V_img.row(F_img(fid,0))*bc(0) + V_img.row(F_img(fid,1))*bc(1) + V_img.row(F_img(fid,2))*bc(2);
        int cid = 0;
        Eigen::Vector3d(
            (V_img.row(F_img(fid,0))-m3).squaredNorm(),
            (V_img.row(F_img(fid,1))-m3).squaredNorm(),
            (V_img.row(F_img(fid,2))-m3).squaredNorm()).minCoeff(&cid);
        const int vid = F_img(fid,cid);
        std::cout << "Source index: " << vid << std:: endl;
        pmm_geodesics_solve(data, V_img, F_img, (Eigen::VectorXi(1,1)<<vid).finished(), D, N_iters);
      }
      // else
      // {
      //   D = Eigen::VectorXd::Zero(V_img.rows());
      //   for(int cid = 0;cid<3;cid++)
      //   {
      //     const int vid = F_img(fid,cid);
      //     Eigen::VectorXd Dc;
      //     pmm_geodesics_solve(data, V_img, F_img, (Eigen::VectorXi(1,1)<<vid).finished(), Dc, N_iters);
      //     D += Dc*bc(cid);
      //   }
      // }
      TIMER_END();
      TIMER_START("Updating distances");
      viewer.data().set_data(D);
      TIMER_END();
      #ifdef MATLAB_DEBUG
        TIMER_START("Making a copy of the distance map");
        GMMDenseColMatrix GMM_D_T(cols, rows);
        for (size_t j = 0; j < rows; ++j) {
          for (size_t i = 0; i < cols; ++i) {
            GMM_D_T(i, j) = D(i + j * cols);
          }
        }
        TIMER_END();
        TIMER_START("Sending distance map to Matlab");
        MatlabGMMDataExchange::SetEngineDenseMatrix("D_T", GMM_D_T);
        TIMER_END();
        TIMER_START("Transposing the distance map in Matlab");
        MatlabInterface::GetEngine().EvalToCout(R"(D = D_T.';)");
        TIMER_END();
      #endif
      #ifdef MATRIX_FILE
        if (is_matrix_file) {
          matrix_file.seekp(matrix_file_D_pos);
          TIMER_START("Writing D matrix");
          try {
            bin_write_arr(matrix_file, D.data(), img_len);
          } catch (const std::exception &e) {
            TIMER_ERROR("Write to matrix file failed");
            std::cerr << "Exception: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
          }
          std::cout << "Written " << img_len * sizeof(D.data()[0]) << " bytes (seek=" << matrix_file.tellp() << ")" << std::endl;
          TIMER_END();
        }
      #endif
      return true;
    }
    return false;
  };
  viewer.callback_mouse_down =
    [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
  {
    if(update())
    {
      down_on_mesh = true;
      return true;
    }
    return false;
  };
  viewer.callback_mouse_move =
    [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
      if(down_on_mesh)
      {
        update();
        return true;
      }
      return false;
    };
  viewer.callback_mouse_up =
    [&down_on_mesh](igl::opengl::glfw::Viewer& viewer, int, int)->bool
  {
    down_on_mesh = false;
    return false;
  };

  // Plot the geometry image
  // igl::opengl::glfw::Viewer viewer;
  TIMER_START("Drawing mesh");
  viewer.data().set_mesh(V_img, F_img);
  TIMER_END();
  // viewer.data().set_uv(V_uv_img);
  TIMER_START("Initializing geodesic distances to zero");
  viewer.data().set_data(Eigen::VectorXd::Zero(V_img.rows()));
  TIMER_END();
  TIMER_START("Setting color map");
  set_colormap(viewer);
  TIMER_END();
  viewer.callback_key_down = &key_down;

  // Enable wireframe
  viewer.data().show_lines = true;

  // Draw checkerboard texture
  viewer.data().show_texture = true;

  // If supplied with the last optional argument of starting source,
  //  initially run the PMM algorithm on it
  if (start_with_source) {
    std::cout << "Running initial PMM with ";
    {
      size_t i, len = start_source.size();
      for (i = 0; i < len - 1; ++i) {
        std::cout << start_source[i] << ' ';
      }
      std::cout << start_source[i];
    }
    std::cout << "..." << std::endl;
    timer.start();
    Eigen::VectorXd D;
    Eigen::Map<Eigen::Matrix<size_t, -1, -1> > start_source_vec(start_source.data(), start_source.size(), 1);
    pmm_geodesics_solve(data, V_img, F_img, start_source_vec, D, N_iters);
    TIMER_END();
    TIMER_START("Updating distances");
    viewer.data().set_data(D);
    TIMER_END();
    #ifdef MATLAB_DEBUG
      TIMER_START("Making a copy of the distance map");
      GMMDenseColMatrix GMM_D_T(cols, rows);
      for (size_t j = 0; j < rows; ++j) {
        for (size_t i = 0; i < cols; ++i) {
          GMM_D_T(i, j) = D(i + j * cols);
        }
      }
      TIMER_END();
      TIMER_START("Sending distance map to Matlab");
      MatlabGMMDataExchange::SetEngineDenseMatrix("D_T", GMM_D_T);
      TIMER_END();
      TIMER_START("Transposing the distance map in Matlab");
      MatlabInterface::GetEngine().EvalToCout(R"(D = D_T.';)");
      TIMER_END();
    #endif
    #ifdef MATRIX_FILE
      if (is_matrix_file) {
        matrix_file.seekp(matrix_file_D_pos);
        TIMER_START("Writing D matrix");
        try {
          bin_write_arr(matrix_file, D.data(), img_len);
        } catch (const std::exception &e) {
          TIMER_ERROR("Write to matrix file failed");
          std::cerr << "Exception: " << e.what() << std::endl;
          exit(EXIT_FAILURE);
        }
        std::cout << "Written " << img_len * sizeof(D.data()[0]) << " bytes (seek=" << matrix_file.tellp() << ")" << std::endl;
        TIMER_END();
      }
    #endif
  }

  // Launch the viewer
  std::cout << "Starting the model viewer!" << std::endl;
  viewer.launch();
  std::cout << "Exiting" << std::endl;

  #ifdef MATRIX_FILE
    if (is_matrix_file) {
      try {
        matrix_file.close();
      } catch (const std::exception &e) {
        std::cerr << "Error: Couldn't close the matrix file" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  #endif
}
