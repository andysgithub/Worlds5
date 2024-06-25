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

__device__ void VectorTrans2(double x, double y, double z, vector5Double* c);
__device__ bool ProcessPoint(float* Modulus, float* Angle, float bailout, vector5Double c);
__device__ bool SamplePoint2(double distance, float* Modulus, float* Angle, float bailout, double xFactor, double yFactor, double zFactor, vector5Double c);
__device__ double FindSurface(double increment, double smoothness, int binarySearchSteps, double currentDistance, double xFactor, double yFactor, double zFactor, float bailout);
__device__ double FindBoundary(double increment, int binarySearchSteps, double currentDistance, float previousAngle,
    double boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
    double xFactor, double yFactor, double zFactor, float bailout);

//extern "C" {
//    __global__ void TraceRayKernel(double xFactor, double yFactor, double zFactor,
//        int* externalPoints, float* modulusValues, float* angles, double* distances);
//}
//extern "C" {
//    __global__ void ProcessPointKernel(float* d_Modulus, float* d_Angle, float bailout, vector5Double* d_c, bool* d_result);
//}
//extern "C" {
//    __global__ void SamplePointKernel(double distance, float* d_Modulus, float* d_Angle, float bailout,
//        double xFactor, double yFactor, double zFactor, vector5Double c, bool* d_result);
//}

#endif // KERNEL_CUH