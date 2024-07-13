#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector_types.h>
#include <algorithm>
#include <cmath>
#include <iostream>
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

static int getGridX(int raysPerLine, int totalLines, int xBlocks) {

    const int threadsPerBlock = 256; // 32 * 8 * 1
    const int warpsPerBlock = threadsPerBlock / 32;
    const int blocksPerSM = 8; // This is a general value, might need tuning
    const int numSMs = 20; // For GTX 1080

    // Calculate total threads needed
    int totalThreadsNeeded = raysPerLine * totalLines;

    // Calculate optimal number of blocks
    int optimalNumBlocks = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

    // Limit the number of blocks to a multiple of the optimal blocks per SM
    int limitedNumBlocks = std::min(optimalNumBlocks, blocksPerSM * numSMs);
    limitedNumBlocks = (limitedNumBlocks + warpsPerBlock - 1) / warpsPerBlock * warpsPerBlock;

    // Return grid X dimension
    return std::min(limitedNumBlocks, (int)((raysPerLine + xBlocks - 1) / xBlocks));
}

extern "C" cudaError_t LaunchProcessRaysKernel(const RayTracingParams* rayParams, const RenderingParams* renderParams,
    int raysPerLine, int totalLines, ProgressCallback callback)
{
    // Allocate device memory
    RayDataTypeIntermediate* d_results;
    cudaError_t cudaStatus = cudaMalloc(&d_results, raysPerLine * totalLines * sizeof(RayDataTypeIntermediate));
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
 
    dim3 blockDim(32, 8, 1);
    dim3 gridDim(
        getGridX(raysPerLine, totalLines, blockDim.x),
        (totalLines + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    ProcessRaysKernel << <gridDim, blockDim >> > (*rayParams, *renderParams, raysPerLine, totalLines, d_results);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {      
        cudaFree(d_results);
        return cudaStatus;
    }

    // Allocate host memory
    RayDataTypeIntermediate* h_results = new RayDataTypeIntermediate[raysPerLine * totalLines];

    // Copy results back to host
    cudaStatus = cudaMemcpy(h_results, d_results, raysPerLine * totalLines * sizeof(RayDataTypeIntermediate), cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) {
        std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
        delete[] h_results;
        cudaFree(d_results);
        return cudaStatus;
    }

    // Process results and call the callback for each ray
    for (int rowCount = 0; rowCount < totalLines; ++rowCount) {
        for (int rayCount = 0; rayCount < raysPerLine; ++rayCount) {
            int index = rowCount * raysPerLine + rayCount;
            if (callback) {
                callback(rayCount, rowCount, &h_results[index]);
            }
        }
    }

    // Free memory
    delete[] h_results;
    cudaFree(d_results);

    return cudaSuccess;
}