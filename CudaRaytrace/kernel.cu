#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdio.h>
#include "inline.cuh"
#include "cuda_interface.h"
#include "RayTracer.cuh"

__constant__ RayTracingParams d_params;

// Host function to initialize the GPU with constant parameters
extern "C" cudaError_t InitializeGPUKernel(const RayTracingParams* params)
{
    // Copy the parameters to the device's constant memory
    return cudaMemcpyToSymbol((const void*)&d_params, (const void*)params, sizeof(RayTracingParams));

    void* d_addr;
    cudaError_t error;

    // Get the address of the symbol in device memory
    error = cudaGetSymbolAddress(&d_addr, (const void*)&d_params);
    if (error != cudaSuccess) {
        return error;
    }

    // Copy the data to the symbol
    error = cudaMemcpy(d_addr, params, sizeof(RayTracingParams), cudaMemcpyHostToDevice);
    return error;
}

extern "C" cudaError_t InitializeTransformMatrix(const float* positionMatrix)
{
    return cudaMemcpyToSymbol(cudaTrans, positionMatrix, sizeof(float) * DimTotal * (DimTotal + 1));
}

extern "C" int launchTraceRayKernel(float XFactor, float YFactor, float ZFactor, int rayPoints,
    int* externalPoints, float* modulusValues, float* angles, float* distances)
{
    // Allocate device memory
    int* d_externalPoints, * d_recordedPointsOut;
    float* d_modulusValues, * d_angles;
    float* d_distances;

    cudaMalloc(&d_externalPoints, rayPoints * sizeof(int));
    cudaMalloc(&d_modulusValues, rayPoints * sizeof(float));
    cudaMalloc(&d_angles, rayPoints * sizeof(float));
    cudaMalloc(&d_distances, rayPoints * sizeof(float));
    cudaMalloc(&d_recordedPointsOut, sizeof(int));

    // Launch kernel
    TraceRayKernel<<<1, 1>>>(XFactor, YFactor, ZFactor, rayPoints,
        d_externalPoints, d_modulusValues, d_angles, d_distances, d_recordedPointsOut);

    // Copy results back to host
    int recordedPoints;
    cudaMemcpy(externalPoints, d_externalPoints, rayPoints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulusValues, d_modulusValues, rayPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(angles, d_angles, rayPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, d_distances, rayPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&recordedPoints, d_recordedPointsOut, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_externalPoints);
    cudaFree(d_modulusValues);
    cudaFree(d_angles);
    cudaFree(d_distances);
    cudaFree(d_recordedPointsOut);

    return recordedPoints;
}