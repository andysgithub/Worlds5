#pragma once

#include <cuda_runtime.h>
#include "Vectors.cuh"
#include "Parameters.h"

// Constant memory declaration
extern __constant__ float cudaTrans[6][DimTotal];

extern __constant__ RayTracingParams d_rayParams;
extern __constant__ RenderingParams d_renderParams;

namespace RayTracer {
    __device__ int TraceRay(float startDistance, Vector3 rayPoint, int rayPoints,
        int* __restrict__ externalPoints, float* __restrict__ modulusValues,
        float* __restrict__ angles, float* __restrict__ distances);

    __device__ float FindSurface(
        float samplingInterval, float surfaceSmoothing, float currentDistance,
        Vector3 rayPoint);

    __device__ float FindBoundary(float currentDistance, float previousAngle,
        bool* externalPoint, float* Modulus, float* Angle,
        Vector3 rayPoint);

    __device__ bool SamplePoint(float distance, float* Modulus, float* Angle, Vector3 rayPoint, Vector5 c);

    __device__ bool SamplePoint(float distance, Vector3 rayPoint, Vector5 c);

    __device__ bool gapFound(float currentDistance, Vector3 rayPoint, Vector5 c);

    __device__ void VectorTrans(Vector3 imagePoint, Vector5* c);

    __device__ float vectorAngle(const Vector5& A, const Vector5& B, const Vector5& C);
}