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

#include <cuda_runtime.h>
#include "utils_cuda.h"

#include "data_shared_path.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>
#include <array>
#include "plf_nanotimer/plf_nanotimer.h"
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include "utils_perf.h"
#include "utils_cuda.h"

#ifdef PRINT_DEVICE_INFO
  bool print_device_info = true;
#endif

#if 0
  #define TIMER_START(MSG) do { \
                               std::cout << MSG "..." << std::endl; timer.start(); \
                           } while(0)
  #define TIMER_END()      do { \
                               std::cout << "Done!\t" << timer.get_elapsed_s() << "s" << std::endl; \
                           } while(0)
  #define TIMER_ERROR(MSG) do { \
                               std::cerr << "Error\t" << timer.get_elapsed_s() << "s" << ": " << MSG << std::endl; \
                           } while(0)
#else
  #define TIMER_START(MSG) perfUtil.meas(MSG);
  #define TIMER_END()      perfUtil.stop();
  #define TIMER_ERROR(MSG) perfUtil.error(MSG)
#endif

#ifndef SCALAR_
  #define SCALAR_
  #if defined(SCALAR_DOUBLE)
    typedef double Scalar;
    typedef double2 Scalar2;
    typedef double3 Scalar3;
    typedef double4 Scalar4;
  #else
    typedef float Scalar;
    typedef float2 Scalar2;
    typedef float3 Scalar3;
    typedef float4 Scalar4;
  #endif
#endif

PerfUtil perfUtil;
PerfCuda perfCuda;

int main(int argc, char *argv[])
{
  cudaError_t stat;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    #ifdef PRINT_DEVICE_INFO
      ("no-device-info,n", "don't print device information")
    #endif
  ;
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  #ifdef PRINT_DEVICE_INFO
    if (vm.count("no-device-info")) {
      print_device_info = false;
    }
  #endif

  if (vm.count("help")) {
    std::cout << desc << std::endl;
  }

  std::cout << std::endl;

  #ifdef PRINT_DEVICE_INFO
    if (print_device_info) {
      int dev_num;
      checkCuda(cudaGetDeviceCount(&dev_num));
      for (int i = 0; i < dev_num; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device number: " << i << std::endl;
        std::cout << "\t" "Device name:" "\t\t\t" << prop.name << std::endl;
        std::cout << "\t" "Memory clock rate:" "\t\t" << prop.memoryClockRate << " KHz" << std::endl;
        std::cout << "\t" "Memory bus width:" "\t\t"<< prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "\t" "Peak memory bandwidth:" "\t\t" <<
          2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0E6 <<
          " GB/s" << std::endl;
        std::cout << "\t" "Compute capability:" "\t\t" << prop.major << "." << prop.minor << std::endl;
        std::cout << "\t" "Multiprocessors count:" "\t\t" << prop.multiProcessorCount << std::endl;
        std::cout << "\t" "Warp size:" "\t\t\t" << prop.warpSize << std::endl;
        std::cout << "\t" "Total global memory:" "\t\t" << (prop.totalGlobalMem / (double)(1<<20)) << " MiB" << std::endl;
        std::cout << "\t" "Total const memory:" "\t\t" << (prop.totalConstMem / (double)(1<<20)) << " MiB" << std::endl;
        std::cout << "\t" "Shared memory per block:" "\t" << (prop.sharedMemPerBlock / (double)(1<<20)) << "MiB" << std::endl;
        std::cout << "\t" "Registers per block (4 bytes):" "\t" << prop.regsPerBlock << std::endl;
        std::cout << "\t" "Max threads per block:" "\t\t" << prop.maxThreadsPerBlock << std::endl;
        std::cout << "\t" "Max block dimensions:" "\t\t";
        {
          auto *arr = prop.maxThreadsDim;
          int i = 0;
          for (i = 0; i < 3; ++i) {
            std::cout << "[" << i << "] " << arr[i] << ", ";
          }
          std::cout << "[" << i << "] " << arr[i] << std::endl;
        }
        std::cout << "\t" "Max grid dimensions:" "\t\t";
        {
          auto *arr = prop.maxGridSize;
          int i = 0;
          for (i = 0; i < 3; ++i) {
            std::cout << "[" << i << "] " << arr[i] << ", ";
          }
          std::cout << "[" << i << "] " << arr[i] << std::endl;
        }
        std::cout << "\t" "Memory copy pitch size:" "\t\t" << prop.memPitch << " bytes" << std::endl;
        std::cout << "\t" "Concurrent copy and execute:" "\t" << prop.deviceOverlap << "(" << ((prop.deviceOverlap) ? (std::string("True")) : (std::string("False"))) << ")" << std::endl;
        std::cout << "\t" "Device can map host memory:" "\t" << prop.canMapHostMemory << "(" << ((prop.canMapHostMemory) ? (std::string("True")) : (std::string("False"))) << ")" << std::endl;
        std::cout << "\t" "Device is integrated:" "\t\t" << prop.integrated << "(" << ((prop.integrated) ? (std::string("Integrated")) : (std::string("Discrete/Dedicated"))) << ")" << std::endl;
        std::cout << "\t" "Kernel timeout enabled:" "\t\t" << prop.kernelExecTimeoutEnabled << "(" << ((prop.kernelExecTimeoutEnabled) ? (std::string("True")) : (std::string("False"))) << ")" << std::endl;
        std::cout << "\t" "Device compute mode:" "\t\t" << prop.computeMode << "(";
        {
          switch (prop.computeMode) {
          case cudaComputeMode::cudaComputeModeDefault:
            std::cout << "cudaComputeModeDefault";
            break;
          case cudaComputeMode::cudaComputeModeExclusive:
            std::cout << "cudaComputeModeExclusive";
            break;
          case cudaComputeMode::cudaComputeModeProhibited:
            std::cout << "cudaComputeModeProhibited";
            break;
          }
          std::cout << ")" << std::endl;
        }
      }
    }
  #endif

  std::cout << std::endl;
}
