#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h> // Include CUDA runtime header for device functions and types
#include "vector5Double.h"

#define DimTotal 5

// Constant memory declaration
__constant__ double cudaTrans[DimTotal][6];

//// Define the vector5Double structure
//struct vector5Double {
//    double coords[5];
//
//    // Method to convert the vector to a 5D array
//    __host__ __device__
//        void toArray(double array[5]) const {
//        array[0] = coords[0];
//        array[1] = coords[1];
//        array[2] = coords[2];
//        array[3] = coords[3];
//        array[4] = coords[4];
//    }
//};

__device__ void VectorTrans(double x, double y, double z, vector5Double* c);
__device__ bool ProcessPoint(float* Modulus, float* Angle, float bailout, vector5Double c);
__device__ bool SamplePoint(double distance, float* Modulus, float* Angle, float bailout, double xFactor, double yFactor, double zFactor, vector5Double c);
__device__ double FindSurface(double increment, double smoothness, int binarySearchSteps, double currentDistance, double xFactor, double yFactor, double zFactor, float bailout);
__device__ double FindBoundary(double increment, int binarySearchSteps, double currentDistance, float previousAngle,
    double boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
    double xFactor, double yFactor, double zFactor, float bailout);

__global__ void TraceRayKernel(double startDistance, double increment, double smoothness, double surfaceThickness,
    double xFactor, double yFactor, double zFactor, float bailout,
    int* externalPoints, float* modulusValues, float* angles, double* distances,
    int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
    int activeIndex);

void launchTraceRayKernel(double startDistance, double increment, double smoothness, double surfaceThickness,
    double XFactor, double YFactor, double ZFactor, float bailout,
    int* externalPoints, float* modulusValues, float* angles, double* distances,
    int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
    int activeIndex);

// Function to copy transformation matrix to device
void copyTransformationMatrixToDevice(double h_Trans[DimTotal][6]);

#endif // KERNEL_CUH