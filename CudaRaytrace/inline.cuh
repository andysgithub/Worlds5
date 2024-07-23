#pragma once

//#include <cuda_runtime.h>
//#include "vectors.cuh"
//
//__device__ __forceinline__ void v_mov(const Vector5 a, Vector5 b) {
//    b[0] = a[0];
//    b[1] = a[1];
//    b[2] = a[2];
//    b[3] = a[3];
//    b[4] = a[4];
//}
//
//__device__ __forceinline__ void v_subm(const Vector5 a, const Vector5 b, Vector5 c) {
//    c[0] = b[0] - a[0];
//    c[1] = b[1] - a[1];
//    c[2] = b[2] - a[2];
//    c[3] = b[3] - a[3];
//    c[4] = b[4] - a[4];
//}
//
//__device__ __forceinline__ float v_mod(const Vector5 a) {
//    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3] + a[4] * a[4];
//}
//
//__device__ __forceinline__ Vector5 v_mandel(Vector5 a, const Vector5 b) {
//    float a0 = a[0];
//    float a1 = a[1];
//    float a2 = a[2];
//    float a3 = a[3];
//    float a4 = a[4];
//
//    a[0] = a0 * a0 - a1 * (a1 - a2 + a3 - a4) + a1 * (a2 - a3 + a4) + b[0];
//    a[1] = 2 * a0 * a1 + a2 * (a2 - a3 + a4) - a2 * (a3 - a4) + b[1];
//    a[2] = 2 * a0 * a2 - a3 * (a3 - a4) + a3 * a4 + b[2];
//    a[3] = 2 * a0 * a3 + a4 * a4 + b[3];
//    a[4] = 2 * a0 * a4 + b[4];
//
//    return a;
//}
