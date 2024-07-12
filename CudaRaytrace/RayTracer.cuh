#pragma once

#include <cuda_runtime.h>
#include "inline.cuh"
#include "vector5Single.h"
#include "cuda_interface.h"

// Constant memory declaration
extern __constant__ float cudaTrans[6][DimTotal];

namespace RayTracer {
    __device__ int TraceRay2(float startDistance, RayTracingParams rayParams,
        float xFactor, float yFactor, float zFactor, int rayPoints,
        int* externalPoints, float* modulusValues, float* angles, float* distances);

    __device__ float FindSurface2(
        float samplingInterval, float surfaceSmoothing, int binarySearchSteps, float currentDistance,
        float xFactor, float yFactor, float zFactor, float bailout);

    __device__ float FindBoundary2(float samplingInterval, int binarySearchSteps, float currentDistance, float previousAngle,
        float boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
        float xFactor, float yFactor, float zFactor, float bailout);

    __device__ bool SamplePoint2(float distance, float* Modulus, float* Angle, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c);

    __device__ bool SamplePoint2(float distance, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c);

    __device__ bool ExternalPoint2(vector5Single c, float bailout);

    __device__ bool ProcessPoint2(float* Modulus, float* Angle, float bailout, vector5Single c);

    __device__ bool gapFound2(float currentDistance, float surfaceThickness, float xFactor, float yFactor, float zFactor, float bailout, vector5Single c);

    __device__ void VectorTrans2(float x, float y, float z, vector5Single* c);

    __device__ float vectorAngle(const vector5Single& A, const vector5Single& B, const vector5Single& C);
}