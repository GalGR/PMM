#pragma once

__global__ void solve_upwards(
    Scalar *D, cudaTextureObject_t V, Scalar *C,
    unsigned tile_width, unsigned tile_height, unsigned tile_pitch, unsigned tile_eff_width,
    unsigned tile_eff_height, unsigned tile_offset, unsigned yoffset,
    unsigned width, unsigned height,
    unsigned D_pitch, unsigned C_pitch
) {
    extern __shared__ Scalar d_shared[];
    // How many elements we go over each stride
    const unsigned xstride = gridDim.x * tile_eff_width;
    // Stride all over the matrix
    for (unsigned xoffset = 0; xoffset < width + tile_offset; xoffset += xstride) {
        const unsigned x = xoffset + blockIdx.x * tile_eff_width + threadIdx.x - tile_offset; // Negative number will result in x > width
        const unsigned y = yoffset;
        // Copy for each tile, 'D' to shared memory 'd_shared'
        for (unsigned i = 0; i < tile_height && x < width; ++i) {
            if (y + i < height) {
                d_shared[threadIdx.x + i * tile_pitch] = D[x + (y + i) * D_pitch];
            }
        }
        // Sync all threads in block after copying to shared memory
        __syncthreads();
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
                if (1 <= x && x <= width - 1 && 1 <= threadIdx.x && threadIdx.x <= tile_width - 1) { // Left triangle -- even rows coefficients
                    Scalar4 ab = *reinterpret_cast<Scalar4*>(&C[(PMM_A_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (2 * (y + i - 1)) * C_pitch]);
                    Scalar4 c  = *reinterpret_cast<Scalar4*>(&C[(PMM_C_OFF + 0) + (x - 1) * PMM_COEFF_PITCH + (2 * (y + i - 1)) * C_pitch]);
                    Scalar d_new = solve_kernel(x_l, x_m,
                        make_Scalar2(d_shared[idx_d_l], d_shared[idx_d_m]),
                        ab.x, make_Scalar2(ab.z, ab.w), c
                    );
                    d_shared[idx_d_0] = fmin(d_new, d_shared[idx_d_0]);
                }
                if (0 <= x && x <= width - 2 && 0 <= threadIdx.x && threadIdx.x <= tile_width - 2) { // Right triangle -- odd rows coefficients
                    Scalar4 ab = *reinterpret_cast<Scalar4*>(&C[(PMM_A_OFF + 0) + x * PMM_COEFF_PITCH + (2 * (y + i - 1) + 1) * C_pitch]);
                    Scalar4 c  = *reinterpret_cast<Scalar4*>(&C[(PMM_C_OFF + 0) + x * PMM_COEFF_PITCH + (2 * (y + i - 1) + 1) * C_pitch]);
                    Scalar d_new = solve_kernel(x_r, x_m,
                        make_Scalar2(d_shared[idx_d_r], d_shared[idx_d_m]),
                        ab.x, make_Scalar2(ab.z, ab.w), c
                    );
                    d_shared[idx_d_0] = fmin(d_new, d_shared[idx_d_0]);
                }
            }
        }
        // Copy for each tile, 'd_shared' to global memory 'D'
        for (unsigned i = 1; i < tile_height && x < width; ++i) {
            if (y + i < height && threadIdx.x >= i && threadIdx.x < i + tile_eff_width) {
                D[x + (y + i) * D_pitch] = d_shared[threadIdx.x + i * tile_pitch];
            }
        }
    }
}