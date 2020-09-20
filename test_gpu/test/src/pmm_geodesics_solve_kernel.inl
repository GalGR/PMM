#pragma once

__device__ Scalar solve_kernel(const Scalar3 x1, const Scalar3 x2, const Scalar2 d, const Scalar a, const Scalar2 b2, const Scalar4 c4
#ifdef CUDA_DEBUG_PRINT
    , const unsigned x, const unsigned y
#endif
) {
    // The Scalar 2 b is a row vector
    // The Scalar 4 c is a col major 2x2 matrix
    const Scalar b = -2.0 * b2.x * d.x + b2.y * d.y;
    const Scalar c = -1.0 + (c4.x * d.x + c4.y * d.y) * d.x + (c4.z * d.x + c4.w * d.y) * d.y;

    const Scalar a_2 = 2.0 * a;
    const Scalar sqroot = sqrt(b * b - 4.0 * a * c);
    const Scalar rhs = sqroot / a_2;
    const Scalar lhs = -b / a_2;
    const Scalar d_quadratic = fmax(lhs - rhs, lhs + rhs);

    const Scalar x1_norm = sqrt(x1.x * x1.x + x1.y * x1.y + x1.z * x1.z);
    const Scalar x2_norm = sqrt(x2.x * x2.x + x2.y * x2.y + x2.z * x2.z);
    const Scalar d_dijkstra = fmin(d.x + x1_norm, d.y + x2_norm);

    const bool cond_inf = isinf(d.x) || isinf(d.y);
    const bool cond_nan = isnan(d_quadratic);
    const bool cond_max = d_quadratic < fmax(d.x, d.y);
    const bool cond_monotonicity = (c4.x * (d.x - d_quadratic) + c4.z * (d.y - d_quadratic) > 0.0)
                                || (c4.y * (d.x - d_quadratic) + c4.w * (d.y - d_quadratic) > 0.0);
    const bool cond = cond_nan || cond_max || cond_monotonicity || cond_inf;

    return cond ? d_dijkstra : d_quadratic;
}