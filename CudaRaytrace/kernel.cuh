#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h> // Include CUDA runtime header for device functions and types
#include "vector5Single.h"

// Constant memory declaration
__constant__ float cudaTrans[6][DimTotal];

__device__ void VectorTrans2(float x, float y, float z, vector5Single* c);
__device__ bool ProcessPoint(float* Modulus, float* Angle, float bailout, vector5Single c);
__device__ bool SamplePoint2(float distance, float* Modulus, float* Angle, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c);
__device__ float FindSurface(float samplingInterval, float surfaceSmoothing, int binarySearchSteps, float currentDistance, float xFactor, float yFactor, float zFactor, float bailout);
__device__ float FindBoundary(float samplingInterval, int binarySearchSteps, float currentDistance, float previousAngle,
    float boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
    float xFactor, float yFactor, float zFactor, float bailout);

#endif // KERNEL_CUH