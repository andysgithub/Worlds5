#pragma once

#include <cuda_runtime.h>
#include "inline.cuh"
#include "vector5Single.h"
#include "cuda_interface.h"

// Constant memory declaration
extern __constant__ float cudaTrans[6][DimTotal];

namespace RayTracer {
    __device__ int TraceRay(float startDistance, const RayTracingParams* __restrict__ rayParams,
        float xFactor, float yFactor, float zFactor, int rayPoints,
        int* __restrict__ externalPoints, float* __restrict__ modulusValues,
        float* __restrict__ angles, float* __restrict__ distances);

    __device__ float FindSurface(
        float samplingInterval, float surfaceSmoothing, int binarySearchSteps, float currentDistance,
        float xFactor, float yFactor, float zFactor, float bailout);

    __device__ float FindBoundary(float samplingInterval, int binarySearchSteps, float currentDistance, float previousAngle,
        float boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
        float xFactor, float yFactor, float zFactor, float bailout);

    __device__ bool SamplePoint(float distance, float* Modulus, float* Angle, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c);

    __device__ bool SamplePoint(float distance, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c);

    __device__ bool ExternalPoint(vector5Single c, float bailout);

    __device__ bool ProcessPoint(float* Modulus, float* Angle, float bailout, vector5Single c);

    __device__ bool gapFound(float currentDistance, float surfaceThickness, float xFactor, float yFactor, float zFactor, float bailout, vector5Single c);

    __device__ void VectorTrans(float x, float y, float z, vector5Single* c);

    __device__ float vectorAngle(const vector5Single& A, const vector5Single& B, const vector5Single& C);
}