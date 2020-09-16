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
#include <cuda_runtime_api.h>
#include "utils_cuda.h"

#include "data_shared_path.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>
#include <cstring>
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

int width, height, depth;
bool is_print_array = false;

__global__ void kernel_tex(cudaTextureObject_t texObj, float *res, int width, int height, int depth, int len) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  int idx = width * height * z + width * y + x;
  if (x < width && y < height && z < depth && idx < len) {
    res[idx] = tex3D<float>(texObj, x + 0.5, y + 0.5, z + 0.5);
  }
}

int main(int argc, char *argv[])
{
  cudaError_t stat;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("width,w", po::value<int>(&width)->default_value(3), "array width")
    ("height,h", po::value<int>(&height)->default_value(3), "array height")
    ("depth,d", po::value<int>(&depth)->default_value(3), "array depth")
    ("print-array,p", "print the host array")
    #ifdef PRINT_DEVICE_INFO
      ("no-device-info,n", "don't print device information")
    #endif
  ;
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("print-array")) {
    is_print_array = true;
  }

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
        std::cout << "\t" "Shared memory per block:" "\t" << (prop.sharedMemPerBlock / (double)(1<<20)) << " MiB" << std::endl;
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
        std::cout << "\t" "Concurrent copy and execute:" "\t" << prop.deviceOverlap << " (" << ((prop.deviceOverlap) ? (std::string("True")) : (std::string("False"))) << ")" << std::endl;
        std::cout << "\t" "Device can map host memory:" "\t" << prop.canMapHostMemory << " (" << ((prop.canMapHostMemory) ? (std::string("True")) : (std::string("False"))) << ")" << std::endl;
        std::cout << "\t" "Device is integrated:" "\t\t" << prop.integrated << " (" << ((prop.integrated) ? (std::string("Integrated")) : (std::string("Discrete/Dedicated"))) << ")" << std::endl;
        std::cout << "\t" "Kernel timeout enabled:" "\t\t" << prop.kernelExecTimeoutEnabled << " (" << ((prop.kernelExecTimeoutEnabled) ? (std::string("True")) : (std::string("False"))) << ")" << std::endl;
        std::cout << "\t" "Device compute mode:" "\t\t" << prop.computeMode << " (";
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

  perfUtil.meas("Measuring the run-time of the program");
  std::cout << std::endl;

  // Create a device 1D array
  float *d_arr;
  perfCuda.meas("Allocating device result array in device memory");
  checkCuda(cudaMalloc(&d_arr, width * height * depth * sizeof(d_arr)));
  perfCuda.stop();

  // Create a host 3D array
  float *h_arr = new float[width * height * depth];
  for (int k = 0; k < depth; ++k) {
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        h_arr[width * height * k + width * j + i] = height * width * k + width * j + i;
      }
    }
  }

  std::cout << std::endl;

  // Print the array
  if (is_print_array) {
    std::cout << "The 3D array on the host:" << std::endl;
    for (int k = 0; k < depth; ++k) {
      std::cout << "depth=" << k << std::endl;
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          std::cout << h_arr[width * height * k + width * j + i];
          if (i < width - 1) std::cout << '\t';
        }
        std::cout << std::endl;
      }
    }
  }
  // std::cout << std::endl << "1D view of the host array:" << std::endl;
  // for (int i = 0; i < width * height * depth; ++i) {
  //   std::cout << h_arr[i];
  // }
  std::cout << std::endl;

  // Create a channel format descriptor -- holds 1 single precision float (can hold up to 4 8-bit floats)
  cudaChannelFormatDesc chanDesc = { sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat };

  // Create a cuda array from that channel format description
  cudaArray_t cu_arr;
  cudaExtent cu_arr_dim = make_cudaExtent(width, height, depth);
  perfCuda.meas("Allocating device 3D CUDA array in device memory");
  checkCuda(cudaMalloc3DArray(&cu_arr, &chanDesc, cu_arr_dim));
  perfCuda.stop();

  // Describe the copying from the host array to the cuda array in the cudaMemcpy3DParms struct
  cudaMemcpy3DParms hostToDevice3DArrayCopyParms;
  std::memset(&hostToDevice3DArrayCopyParms, 0, sizeof(hostToDevice3DArrayCopyParms));
  hostToDevice3DArrayCopyParms.srcPtr = make_cudaPitchedPtr(h_arr, width * sizeof(h_arr[0]), width, height); // The third dimension is not mentioned here
  hostToDevice3DArrayCopyParms.dstArray = cu_arr; // Copying to the device cuda array
  hostToDevice3DArrayCopyParms.extent = make_cudaExtent(width, height, depth); // Here the third dimension is mentioned
  hostToDevice3DArrayCopyParms.kind = cudaMemcpyHostToDevice; // Copying from the host to the device

  // Copy the data from the host array to the cuda array
  perfCuda.meas("Copying host 3D array to device CUDA array");
  checkCuda(cudaMemcpy3D(&hostToDevice3DArrayCopyParms));
  perfCuda.stop();

  // Create cuda resource descriptor for the texture -- the texture is made up from cuda array
  cudaResourceDesc resDesc;
  std::memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cu_arr;

  // Create cuda texture descriptor for the texture -- the texture's properties (how it should be sampled and such)
  cudaTextureDesc texDesc;
  std::memset(&texDesc, 0, sizeof(texDesc));
  for (int i = 0; i < 3; ++i) {
    texDesc.addressMode[i] = cudaAddressModeClamp; // Clamp the coordinates for each of the dimensions
  }
  texDesc.filterMode = cudaFilterModePoint; // Don't interpolate the data between points
  texDesc.readMode = cudaReadModeElementType; // Don't change the type of integers to floats (doesn't really apply here anyway)
  texDesc.normalizedCoords = false; // Texture coordinates are from 0 to LENGTH, and not from 0 to 1

  // Create the texture
  cudaTextureObject_t texObj;
  perfCuda.meas("Binding device 3D CUDA array to texture memory");
  checkCuda(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
  perfCuda.stop();

  // Create dim3 objects for the kernel call
  dim3 dimBlock(16, 16, 16); // Fixed size 3D tiles
  dim3 dimGrid( // Create as many 3D tiles as necessary by dividing the dimensions (width,height,depth) by the fixed size block (rounded up)
    (width  + dimBlock.x - 1) / dimBlock.x, // Round up by adding the (divisor - 1) to the dividend
    (height + dimBlock.y - 1) / dimBlock.y,
    (depth  + dimBlock.z - 1) / dimBlock.z
  );

  std::cout << std::endl;

  // Call the texture kernel
  perfCuda.meas("Executing the texture kernel");
  kernel_tex<<<dimGrid, dimBlock>>>(texObj, d_arr, width, height, depth, width * height * depth);

  // Synchronize with the device
  checkAsyncCuda(cudaDeviceSynchronize());
  perfCuda.stop();

  std::cout << std::endl;

  // Copy the device result array to the host
  float *h_res_arr = new float[width * height * depth];
  perfCuda.meas("Copying the device result array to the host");
  checkCuda(cudaMemcpy(h_res_arr, d_arr, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost));
  perfCuda.stop();

  std::cout << std::endl;

  // Compare h_arr to d_arr
  bool is_found_error = false;
  for (int i = 0; i < width * height * depth; ++i) {
    float err = h_res_arr[i] - h_arr[i];
    if (err > 0.0001) {
      std::cout << "error=" << err << ": h_res_arr[" << i << "]=" << h_res_arr[i] << ", h_arr[" << i << "]=" << h_arr[i] << std::endl;
    }
  }
  if (!is_found_error) {
    std::cout << "No error found!" << std::endl;
  }

  std::cout << std::endl;

  // Free the device arrays
  perfCuda.meas("Destroying the texture object");
  checkCuda(cudaDestroyTextureObject(texObj));
  perfCuda.stop();
  perfCuda.meas("Freeing the CUDA array");
  checkCuda(cudaFreeArray(cu_arr));
  perfCuda.stop();
  perfCuda.meas("Freeing the result array");
  checkCuda(cudaFree(d_arr));
  perfCuda.stop();

  std::cout << std::endl;
  perfUtil.stop();

  delete[] h_arr;
  delete[] h_res_arr;

  return 0;
}
