#ifndef INLINE_FUNCTIONS_CUH
#define INLINE_FUNCTIONS_CUH

#include <cuda_runtime.h>

#define DimTotal 5

__device__ inline void v_mov(const float* a, float* b) {
    for (int i = 0; i < DimTotal; ++i) {
        b[i] = a[i];
    }
}

__device__ inline void v_mul(float* a, const float* b) {
    float b0 = b[0];
    a[0] = a[0] * b0 - a[1] * (b[1] - b[2] + b[3] - b[4]) + b[1] * (a[2] - a[3] + a[4]);
    a[1] = a[0] * b[1] + a[1] * b0 + a[2] * (b[2] - b[3] + b[4]) - b[2] * (a[3] - a[4]);
    a[2] = a[0] * b[2] + a[2] * b0 - a[3] * (b[3] - b[4]) + b[3] * a[4];
    a[3] = a[0] * b[3] + a[3] * b0 + a[4] * b[4];
    a[4] = a[0] * b[4] + a[4] * b0;
}

__device__ inline void v_mulm(const float* a, const float* b, float* c) {
    c[0] = a[0] * b[0] - a[1] * (b[1] - b[2] + b[3] - b[4]) + b[1] * (a[2] - a[3] + a[4]);
    c[1] = a[0] * b[1] + a[1] * b[0] + a[2] * (b[2] - b[3] + b[4]) - b[2] * (a[3] - a[4]);
    c[2] = a[0] * b[2] + a[2] * b[0] - a[3] * (b[3] - b[4]) + b[3] * a[4];
    c[3] = a[0] * b[3] + a[3] * b[0] + a[4] * b[4];
    c[4] = a[0] * b[4] + a[4] * b[0];
}

__device__ inline void v_mulc(float* a, float b) {
    for (int i = 0; i < DimTotal; ++i) {
        a[i] *= b;
    }
}

__device__ inline void v_add(const float* a, float* b) {
    for (int i = 0; i < DimTotal; ++i) {
        b[i] += a[i];
    }
}

__device__ inline void v_addm(const float* a, const float* b, float* c) {
    for (int i = 0; i < DimTotal; ++i) {
        c[i] = a[i] + b[i];
    }
}

__device__ inline void v_subm(const float* a, const float* b, float* c) {
    for (int i = 0; i < DimTotal; ++i) {
        c[i] = b[i] - a[i];
    }
}

__device__ inline void v_clr(float* a) {
    for (int i = 0; i < DimTotal; ++i) {
        a[i] = 0;
    }
}

__device__ inline float v_mod(const float* a) {
    float sum = 0;
    for (int i = 0; i < DimTotal; ++i) {
        sum += a[i] * a[i];
    }
    return sum;
}

__device__ inline void v_mandel(float* a, const float* b) {
    float a0 = a[0];
    a[0] = a0 * a0 - a[1] * (a[1] - a[2] + a[3] - a[4]) + a[1] * (a[2] - a[3] + a[4]) + b[0];
    a[1] = 2 * a0 * a[1] + a[2] * (a[2] - a[3] + a[4]) - a[2] * (a[3] - a[4]) + b[1];
    a[2] = 2 * a0 * a[2] - a[3] * (a[3] - a[4]) + a[3] * a[4] + b[2];
    a[3] = 2 * a0 * a[3] + a[4] * a[4] + b[3];
    a[4] = 2 * a0 * a[4] + b[4];
}

#endif // INLINE_FUNCTIONS_CUH
