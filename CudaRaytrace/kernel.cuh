#pragma once

#include <cuda_runtime.h> // Include CUDA runtime header for device functions and types
#include "inline.cuh" // Include the inline functions

#define DimTotal 5

// Constant memory declaration
__constant__ double m_Trans[DimTotal][6];

struct vector5Double {
    double coords[DimTotal];
};

__device__ void VectorTrans(double x, double y, double z, vector5Double* c);
__device__ bool ProcessPoint(float* Modulus, float* Angle, vector5Double c);
__device__ bool SamplePoint(double distance, float* Modulus, float* Angle, double xFactor, double yFactor, double zFactor, vector5Double c);
__device__ double FindSurface(double increment, double smoothness, int binarySearchSteps, double currentDistance, double xFactor, double yFactor, double zFactor, float bailout);
__device__ double FindBoundary(double increment, int binarySearchSteps, double currentDistance, float previousAngle,
    double boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
    double xFactor, double yFactor, double zFactor, float bailout);
__global__ void TraceRayKernel(double startDistance, double increment, double smoothness, double surfaceThickness,
    double xFactor, double yFactor, double zFactor, float bailout,
    int* externalPoints, float* modulusValues, float* angles, double* distances,
    int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
    int activeIndex);

// Function to copy transformation matrix to device
void copyTransformationMatrixToDevice(double h_Trans[DimTotal][6]);
