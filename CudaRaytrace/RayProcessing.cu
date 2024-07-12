#include <cuda_runtime.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <cmath>
#include "RayProcessing.h"
#include "TracedRay.h" 
//#include "Clipping.h" 

typedef void(__stdcall* ProgressCallback)(int rayCount, int rowCount, RayDataTypeIntermediate* rayData);

const float DEG_TO_RAD = 0.0174532925F;

__device__ RayDataTypeIntermediate ProcessRay(RayTracingParams rayParams, RenderingParams renderParams, int rayCountX, int rayCountY)
{
    float latitude = rayParams.latitudeStart - rayCountY * rayParams.angularResolution;
    float longitude = rayParams.longitudeStart - rayCountX * rayParams.angularResolution;

    float latRadians = latitude * DEG_TO_RAD;
    float longRadians = longitude * DEG_TO_RAD;

    float xFactor = cosf(latRadians) * sinf(-longRadians);
    float yFactor = sinf(latRadians);
    float zFactor = cosf(latRadians) * cosf(-longRadians);

    float startDistance = rayParams.sphereRadius;

    //if (rayParams.useClipping) {
    //    float distance = CalculateDistance(latRadians, longRadians, rayParams.clippingAxes, rayParams.clippingOffset);
    //    if (distance > startDistance) startDistance = distance;
    //}

    int externalPoints[MAX_POINTS];
    float modulusValues[MAX_POINTS];
    float angleValues[MAX_POINTS];
    float distanceValues[MAX_POINTS];

    int points = TraceRay2(startDistance, rayParams,
        xFactor, yFactor, zFactor,
        externalPoints, modulusValues, angleValues, distanceValues);

    // Create and return a RayDataType directly
    RayDataType result;
    result.BoundaryTotal = points;
    result.ArraySize = points;

    // Copy arrays element by element
    for (int i = 0; i < points && i < MAX_POINTS; ++i) {
        result.ExternalPoints[i] = externalPoints[i];
        result.ModulusValues[i] = modulusValues[i];
        result.AngleValues[i] = angleValues[i];
        result.DistanceValues[i] = distanceValues[i];
    }

    return ConvertToIntermediate(result, MAX_POINTS);
}

// CUDA kernel function
__global__ void ProcessRayKernel(RayTracingParams rayParams, RenderingParams renderParams,
    int raysPerLine, int totalLines, RayDataTypeIntermediate* results)
{
    int rayCountX = blockIdx.x * blockDim.x + threadIdx.x;
    int rayCountY = blockIdx.y * blockDim.y + threadIdx.y;

    if (rayCountX >= raysPerLine || rayCountY >= totalLines)
        return;

    RayDataTypeIntermediate rayData = ProcessRay(rayParams, renderParams, rayCountX, rayCountY);

    // Store results in global memory
    int index = rayCountY * raysPerLine + rayCountX;
    results[index] = rayData;
}

extern "C" __declspec(dllexport) void ProcessRays(RayTracingParams rayParams, RenderingParams renderParams,
    int raysPerLine, int totalLines, ProgressCallback callback)
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

    // Call callback function for each result
    for (int rayCountY = 0; rayCountY < totalLines; ++rayCountY) {
        for (int rayCountX = 0; rayCountX < raysPerLine; ++rayCountX) {
            int index = rayCountY * raysPerLine + rayCountX;
            if (callback) {
                callback(rayCountX, rayCountY, &h_results[index]);
            }
        }
    }

    // Free memory
    delete[] h_results;
    cudaFree(d_results);
}

__device__ RayDataTypeIntermediate ConvertToIntermediate(const RayDataType& original, int maxSize) {
    RayDataTypeIntermediate result;
    result.ArraySize = original.ArraySize;
    result.BoundaryTotal = original.BoundaryTotal;

    // Assuming RayDataTypeIntermediate has been modified to use fixed-size arrays
    for (int i = 0; i < result.ArraySize && i < maxSize; ++i) {
        result.ExternalPoints[i] = original.ExternalPoints[i];
        result.ModulusValues[i] = original.ModulusValues[i];
        result.AngleValues[i] = original.AngleValues[i];
        result.DistanceValues[i] = original.DistanceValues[i];
    }

    return result;
}
