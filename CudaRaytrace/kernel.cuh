#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h> // Include CUDA runtime header for device functions and types
#include "vector5Single.h"

#define DimTotal 5

// Constant memory declaration
__constant__ float cudaTrans[DimTotal][6];

//// Define the vector5Single structure
//struct vector5Single {
//    float coords[5];
//
//    // Method to convert the vector to a 5D array
//    __host__ __device__
//        void toArray(float array[5]) const {
//        array[0] = coords[0];
//        array[1] = coords[1];
//        array[2] = coords[2];
//        array[3] = coords[3];
//        array[4] = coords[4];
//    }
//};

__device__ void VectorTrans2(float x, float y, float z, vector5Single* c);
__device__ bool ProcessPoint(float* Modulus, float* Angle, float bailout, vector5Single c);
__device__ bool SamplePoint2(float distance, float* Modulus, float* Angle, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c);
__device__ float FindSurface(float increment, float smoothness, int binarySearchSteps, float currentDistance, float xFactor, float yFactor, float zFactor, float bailout);
__device__ float FindBoundary(float increment, int binarySearchSteps, float currentDistance, float previousAngle,
    float boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
    float xFactor, float yFactor, float zFactor, float bailout);

//extern "C" {
//    __global__ void TraceRayKernel(float xFactor, float yFactor, float zFactor,
//        int* externalPoints, float* modulusValues, float* angles, float* distances);
//}
//extern "C" {
//    __global__ void ProcessPointKernel(float* d_Modulus, float* d_Angle, float bailout, vector5Single* d_c, bool* d_result);
//}
//extern "C" {
//    __global__ void SamplePointKernel(float distance, float* d_Modulus, float* d_Angle, float bailout,
//        float xFactor, float yFactor, float zFactor, vector5Single c, bool* d_result);
//}

#endif // KERNEL_CUH