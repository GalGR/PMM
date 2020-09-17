cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

# Matlab
option(MATLAB_DEBUG                  "Use Matlab debug"   OFF)
if(MATLAB_DEBUG)
    add_definitions(-DMATLAB_DEBUG)
    find_package(Matlab COMPONENTS ENG_LIBRARY MAT_LIBRARY MX_LIBRARY)
    if(Matlab_FOUND)
        include_directories(${Matlab_INCLUDE_DIRS})
    endif()
    file(GLOB MATLAB_UTILS_SRC_FILES src/MatlabUtils/*.cpp)
    add_library(MatlabUtilsLib STATIC ${MATLAB_UTILS_SRC_FILES})
    set(MATLAB_UTILS_LIB MatlabUtilsLib)
endif()

# Matrix
option(MATRIX_FILE                   "Write matrix file"  OFF)
if(MATRIX_FILE)
    add_definitions(-DMATRIX_FILE)
endif()

# Device info
option(DEVICE_INFO             "Query the device for information" ON)
if(DEVICE_INFO)
    add_definitions(-DDEVICE_INFO)
endif()

# Scalar type
option(SCALAR_DOUBLE                 "Use double precision floating point numbers" OFF)
if(SCALAR_DOUBLE)
    add_definitions(-DSCALAR_DOUBLE)
endif()

# CUDA
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(LIBIGL REQUIRED QUIET)
# add_subdirectory(cmake)

# Add boost
find_package(Boost COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

# compile the source files
include_directories(${Matlab_INCLUDE_DIRS})
file(GLOB SRCFILES src/*.cpp)
file(GLOB CUDAFILES src/*.cu)
add_executable(${PROJECT_NAME} ${CUDAFILES} ${SRCFILES})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw ${Boost_LIBRARIES} ${CUDART_LIBRARY} ${MATLAB_UTILS_LIB} ${Matlab_LIBRARIES})

# Ignore deprecation warnings
if(NOT MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
endif()