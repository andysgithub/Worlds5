#pragma once

#include <cuda_runtime.h>
#include "vector5Single.h"

__device__ __forceinline__ void v_mov(const float* __restrict__ a, float* __restrict__ b) {
    b[0] = a[0];
    b[1] = a[1];
    b[2] = a[2];
    b[3] = a[3];
    b[4] = a[4];
}

__device__ __forceinline__ void v_subm(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    c[0] = b[0] - a[0];
    c[1] = b[1] - a[1];
    c[2] = b[2] - a[2];
    c[3] = b[3] - a[3];
    c[4] = b[4] - a[4];
}

__device__ __forceinline__ float v_mod(const float* __restrict__ a) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3] + a[4] * a[4];
}

__device__ __forceinline__ void v_mandel(float* __restrict__ a, const float* __restrict__ b) {
    float a0 = a[0];
    float a1 = a[1];
    float a2 = a[2];
    float a3 = a[3];
    float a4 = a[4];

    a[0] = a0 * a0 - a1 * (a1 - a2 + a3 - a4) + a1 * (a2 - a3 + a4) + b[0];
    a[1] = 2 * a0 * a1 + a2 * (a2 - a3 + a4) - a2 * (a3 - a4) + b[1];
    a[2] = 2 * a0 * a2 - a3 * (a3 - a4) + a3 * a4 + b[2];
    a[3] = 2 * a0 * a3 + a4 * a4 + b[3];
    a[4] = 2 * a0 * a4 + b[4];
}
