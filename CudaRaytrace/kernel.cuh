#pragma once

#include <cuda_runtime.h>
#include "vector5Single.h"

__device__ bool ProcessPoint(float* Modulus, float* Angle, float bailout, vector5Single c);

__device__ float FindSurface(
    float samplingInterval, float surfaceSmoothing, int binarySearchSteps, float currentDistance, 
    float xFactor, float yFactor, float zFactor, float bailout);

__device__ float FindBoundary(float samplingInterval, int binarySearchSteps, float currentDistance,
    float previousAngle, float boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
    float xFactor, float yFactor, float zFactor, float bailout);
