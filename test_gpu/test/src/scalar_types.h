#pragma once

#if defined(SCALAR_DOUBLE)
    typedef double Scalar;
    #ifdef __CUDACC__
        typedef double2 Scalar2;
        typedef double3 Scalar3;
        typedef double4 Scalar4;
        #define make_Scalar2(X, Y)       make_double2(X, Y)
        #define make_Scalar3(X, Y, Z)    make_double3(X, Y, Z)
        #define make_Scalar4(X, Y, Z, W) make_double4(X, Y, Z, W)
    #endif
#else
    typedef float Scalar;
    #ifdef __CUDACC__
        typedef float2 Scalar2;
        typedef float3 Scalar3;
        typedef float4 Scalar4;
        #define make_Scalar2(X, Y)       make_float2(X, Y)
        #define make_Scalar3(X, Y, Z)    make_float3(X, Y, Z)
        #define make_Scalar4(X, Y, Z, W) make_float4(X, Y, Z, W)
    #endif
#endif

// Currently CUDA textures can only be of type 'float'
typedef float TexScalar;
#ifdef __CUDACC__
    #define make_TexScalar2(X, Y)       make_float2(X, Y)
    #define make_TexScalar3(X, Y, Z)    make_float3(X, Y, Z)
    #define make_TexScalar4(X, Y, Z, W) make_float4(X, Y, Z, W)
#endif
#define cudaChannelFormatKindTexScalar cudaChannelFormatKindFloat