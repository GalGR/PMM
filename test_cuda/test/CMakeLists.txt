cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

# Device info
option(PRINT_DEVICE_INFO             "Print the device info" ON)
if(PRINT_DEVICE_INFO)
    add_definitions(-DPRINT_DEVICE_INFO)
endif()

# Scalar type
option(SCALAR_DOUBLE                 "Use double precision floating point numbers" OFF)
if(SCALAR_DOUBLE)
    add_definitions(-DSCALAR_DOUBLE)
endif()

# Cuda
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(LIBIGL REQUIRED QUIET)
# add_subdirectory(cmake)

# Add boost
find_package(Boost COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

# compile the source files
file(GLOB SRCFILES src/*.cpp)
file(GLOB CUDAFILES src/*.cu)
add_executable(${PROJECT_NAME} ${CUDAFILES} ${SRCFILES})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw ${Boost_LIBRARIES} ${CUDART_LIBRARY})

# Ignore deprecation warnings
if(NOT MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
endif()