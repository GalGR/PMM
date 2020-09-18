#pragma once

__global__ void solve_upwards(Scalar *D, cudaTextureObject_t V, Scalar *C, unsigned tile_width, unsigned tile_height, unsigned tile_pitch, unsigned tile_eff_width, unsigned tile_eff_height, unsigned tile_offset, unsigned width, unsigned height, unsigned D_pitch, unsigned C_pitch) {
    extern __shared__ Scalar d_shared[];
    #ifdef CUDA_DEBUG_PRINT
        const unsigned idx_x = threadIdx.x;
        const unsigned block_x = blockIdx.x;
        const unsigned block_dim = blockDim.x;
        const unsigned grid_dim = gridDim.x;
    #endif
    // How many elements we go over each stride
    const unsigned xstride = gridDim.x * tile_eff_width;
    const unsigned ystride = tile_eff_height;
    // Stride all over the matrix
    for (unsigned yoffset = 0; yoffset < height - 1; yoffset += ystride) {
        for (unsigned xoffset = 0; xoffset < width; xoffset += xstride) {
            const unsigned x = xoffset + blockIdx.x * tile_eff_width + threadIdx.x - tile_offset; // Negative number will result in x > width
            const unsigned y = yoffset;
            // Copy for each tile, 'D' to shared memory 'd_shared'
            for (unsigned i = 0; i < tile_height && x < width; ++i) {
                if (y + i < height) {
                    #ifdef CUDA_DEBUG_PRINT
                        printf("TILE: (x,y)=(%d,%d), D[%d]=%f, Shared[%d]=%f\n",
                        x, y,
                        x + (y + i) * D_pitch, D[x + (y + i) * D_pitch],
                        threadIdx.x + i * tile_pitch, d_shared[threadIdx.x + i * tile_pitch]);
                    #endif
                    d_shared[threadIdx.x + i * tile_pitch] = D[x + (y + i) * D_pitch];
                }
            }
            // Sync all threads in block after copying to shared memory
            __syncthreads();
            #ifdef CUDA_DEBUG_PRINT
                if (x < width && y < height) {
                    printf("IN: gridDim %d, blockDim %d, block %d, thread %d: (x,y)=(%d,%d), stride=(%d,%d), offset=(%d,%d), tile_width=%d, tile_height=%d, tile_pitch=%d, width=%d, height=%d, D_pitch=%d, C_pitch=%d\nShared[0][%d]=%f, Shared[1][%d]=%f, Shared[tile_height-1][%d]=%f, Shared[tile_height-2][%d]=%f\n",
                    grid_dim, block_dim, block_x, idx_x, x, y, xstride, ystride, xoffset, yoffset, tile_width, tile_height, tile_pitch, width, height, D_pitch, C_pitch,
                    idx_x, d_shared[idx_x], idx_x, d_shared[idx_x + tile_pitch],
                    idx_x, d_shared[idx_x + tile_pitch * (tile_height - 1)], idx_x, d_shared[idx_x + tile_pitch * (tile_height - 2)]);
                }
            #endif
            // C is organized as such: right, (left, right), left
            for (unsigned i = 1; i < tile_height && x < width; ++i) {
                if (y + i < height) {
                    const unsigned idx_d_0 = (threadIdx.x + 0) + (i + 0) * tile_pitch;
                    const unsigned idx_d_l = (threadIdx.x - 1) + (i - 1) * tile_pitch;
                    const unsigned idx_d_m = (threadIdx.x + 0) + (i - 1) * tile_pitch;
                    const unsigned idx_d_r = (threadIdx.x + 1) + (i - 1) * tile_pitch;
                    TexScalar v_x_l = tex3D<TexScalar>(V, (x - 1) + 0.5, (y + i - 1) + 0.5, 0 + 0.5);
                    TexScalar v_x_m = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i - 1) + 0.5, 0 + 0.5);
                    TexScalar v_x_r = tex3D<TexScalar>(V, (x + 1) + 0.5, (y + i - 1) + 0.5, 0 + 0.5);
                    TexScalar v_x_0 = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i + 0) + 0.5, 0 + 0.5);
                    TexScalar v_y_l = tex3D<TexScalar>(V, (x - 1) + 0.5, (y + i - 1) + 0.5, 1 + 0.5);
                    TexScalar v_y_m = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i - 1) + 0.5, 1 + 0.5);
                    TexScalar v_y_r = tex3D<TexScalar>(V, (x + 1) + 0.5, (y + i - 1) + 0.5, 1 + 0.5);
                    TexScalar v_y_0 = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i + 0) + 0.5, 1 + 0.5);
                    TexScalar v_z_l = tex3D<TexScalar>(V, (x - 1) + 0.5, (y + i - 1) + 0.5, 2 + 0.5);
                    TexScalar v_z_m = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i - 1) + 0.5, 2 + 0.5);
                    TexScalar v_z_r = tex3D<TexScalar>(V, (x + 1) + 0.5, (y + i - 1) + 0.5, 2 + 0.5);
                    TexScalar v_z_0 = tex3D<TexScalar>(V, (x + 0) + 0.5, (y + i + 0) + 0.5, 2 + 0.5);
                    Scalar3 x_l = make_Scalar3(v_x_l - v_x_0, v_y_l - v_y_0, v_z_l - v_z_0);
                    Scalar3 x_m = make_Scalar3(v_x_m - v_x_0, v_y_m - v_y_0, v_z_m - v_z_0);
                    Scalar3 x_r = make_Scalar3(v_x_r - v_x_0, v_y_r - v_y_0, v_z_r - v_z_0);
                    if (x >= < 1 && threadIdx.x > 0 && threadIdx.x < tile_width - 1) { // Left triangle -- even rows coefficients
                        // Scalar  a  = *reinterpret_cast<Scalar *>(&C[(PMM_A_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (y + i) * C_pitch * 2]);
                        // Scalar2 b  = *reinterpret_cast<Scalar2*>(&C[(PMM_B_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (y + i) * C_pitch * 2]);
                        Scalar4 ab = *reinterpret_cast<Scalar4*>(&C[(PMM_A_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (y + i) * C_pitch * 2]);
                        Scalar4 c  = *reinterpret_cast<Scalar4*>(&C[(PMM_C_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (y + i) * C_pitch * 2]);
                        // Scalar d_new = solve(x_l, x_m, d_l, d_m, a, b, c);
                        Scalar d_new = solve_kernel(x_l, x_m,
                            make_Scalar2(d_shared[idx_d_l], d_shared[idx_d_m]),
                            ab.x, make_Scalar2(ab.z, ab.w), c
                        #ifdef CUDA_DEBUG_PRINT
                            , x, y
                        #endif
                        );
                        #ifdef CUDA_DEBUG_PRINT
                            printf("LEFT: grid dim %d, block dim %d, block %d, thread %d: (x,y)=(%d,%d), d_old=%f, d_new=%f\n",
                            grid_dim, block_dim, block_x, idx_x, x, y, d_shared[idx_d_0], d_new);
                        #endif
                        d_shared[idx_d_0] = fmin(d_new, d_shared[idx_d_0]);
                    }
                    if (x <= width - 1 && threadIdx.x > 0 && threadIdx.x < tile_width - 1) { // Right triangle -- odd rows coefficients
                        // Scalar  a  = *reinterpret_cast<Scalar *>(&C[(PMM_A_OFF + 0) + x * PMM_COEFF_PITCH + (y + i) * C_pitch * 2 + C_pitch]);
                        // Scalar2 b  = *reinterpret_cast<Scalar2*>(&C[(PMM_B_OFF + 0) + x * PMM_COEFF_PITCH + (y + i) * C_pitch * 2 + C_pitch]);
                        Scalar4 ab = *reinterpret_cast<Scalar4*>(&C[(PMM_A_OFF + 0) + x * PMM_COEFF_PITCH + (y + i) * C_pitch * 2 + C_pitch]);
                        Scalar4 c  = *reinterpret_cast<Scalar4*>(&C[(PMM_C_OFF + 0) + x * PMM_COEFF_PITCH + (y + i) * C_pitch * 2 + C_pitch]);
                        // Scalar d_new = solve(x_m, x_r, d_m, d_r, a, b, c);
                        Scalar d_new = solve_kernel(x_m, x_r,
                            make_Scalar2(d_shared[idx_d_m], d_shared[idx_d_r]),
                            ab.x, make_Scalar2(ab.z, ab.w), c
                        #ifdef CUDA_DEBUG_PRINT
                            , x, y
                        #endif
                        );
                        #ifdef CUDA_DEBUG_PRINT
                            printf("RIGHT: grid dim %d, block dim %d, block %d, thread %d: (x,y)=(%d,%d), d_old=%f, d_new=%f\n",
                            grid_dim, block_dim, block_x, idx_x, x, y, d_shared[idx_d_0], d_new);
                        #endif
                        d_shared[idx_d_0] = fmin(d_new, d_shared[idx_d_0]);
                    }
                }
            }
            // Sync all threads after they calculated the distances in shared memory
            __syncthreads();
            #ifdef CUDA_DEBUG_PRINT
                if (x < width && y < height) {
                    printf("OUT: gridDim %d, blockDim %d, block %d, thread %d: (x,y)=(%d,%d), stride=(%d,%d), offset=(%d,%d), tile_width=%d, tile_height=%d, tile_pitch=%d, width=%d, height=%d, D_pitch=%d, C_pitch=%d\nShared[0][%d]=%f, Shared[1][%d]=%f, Shared[tile_height-1][%d]=%f, Shared[tile_height-2][%d]=%f\n",
                    grid_dim, block_dim, block_x, idx_x, x, y, xstride, ystride, xoffset, yoffset, tile_width, tile_height, tile_pitch, width, height, D_pitch, C_pitch,
                    idx_x, d_shared[idx_x], idx_x, d_shared[idx_x + tile_pitch],
                    idx_x, d_shared[idx_x + tile_pitch * (tile_height - 1)], idx_x, d_shared[idx_x + tile_pitch * (tile_height - 2)]);
                }
            #endif
            // Copy for each tile, 'd_shared' to global memory 'D'
            for (unsigned i = 1; i < tile_height && x < width; ++i) {
                if (y + i < height && threadIdx.x >= i && threadIdx.x < i + tile_eff_width) {
                    D[x + (y + i) * D_pitch] = d_shared[threadIdx.x + i * tile_pitch];
                }
            }
            // Sync all threads after copying to global memory
            __syncthreads();
        }
    }
}