#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdio.h>
#include "inline.cuh"
#include "cuda_interface.h"
#include "RayTracer.cuh"
#include "RayProcessing.cuh"

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

extern "C" void launchProcessRayKernel(RayTracingParams rayParams, RenderingParams renderParams,
    int raysPerLine, int totalLines)
{
    // Allocate device memory
    RayDataTypeIntermediate* d_results;
    cudaMalloc(&d_results, raysPerLine * totalLines * sizeof(RayDataTypeIntermediate));

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((raysPerLine + blockDim.x - 1) / blockDim.x,
        (totalLines + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    ProcessRayKernel << <gridDim, blockDim >> > (rayParams, renderParams, raysPerLine, totalLines, d_results);

    // Allocate host memory and copy results back
    RayDataTypeIntermediate* h_results = new RayDataTypeIntermediate[raysPerLine * totalLines];
    cudaMemcpy(h_results, d_results, raysPerLine * totalLines * sizeof(RayDataTypeIntermediate), cudaMemcpyDeviceToHost);

    // Free memory
    delete[] h_results;
    cudaFree(d_results);
}