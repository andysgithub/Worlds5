#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <vector>
#include <cmath>
#include <stdio.h>
#include "cuda_interface.h"
#include "RayTracer.cuh"
#include "RayProcessing.cuh"
//#include "Clipping.h" 

__device__ const float DEG_TO_RAD = 0.0174532925F;

__device__ void FillIntermediateResult(
    int* externalPoints, float* modulusValues, float* angleValues, float* distanceValues, 
    int points, int maxPoints, RayDataTypeIntermediate* result)
{
    result->ArraySize = (points < maxPoints) ? points : maxPoints;
    result->BoundaryTotal = points;

    for (int i = 0; i < result->ArraySize; ++i) {
        result->ExternalPoints[i] = externalPoints[i];
        result->ModulusValues[i] = modulusValues[i];
        result->AngleValues[i] = angleValues[i];
        result->DistanceValues[i] = distanceValues[i];
    }
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

__device__ void ProcessRayKernel(
    RayTracingParams rayParams, RenderingParams renderParams, int rayCountX, int rayCountY, RayDataTypeIntermediate* result)
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

    int points = RayTracer::TraceRay2(startDistance, rayParams,
        xFactor, yFactor, zFactor, (int)MAX_POINTS,
        externalPoints, modulusValues, angleValues, distanceValues);

    //// Create and return a RayDataType directly
    //RayDataType result;
    //result.BoundaryTotal = points;
    //result.ArraySize = points;

    //// Copy arrays element by element
    //for (int i = 0; i < points && i < MAX_POINTS; ++i) {
    //    result.ExternalPoints[i] = externalPoints[i];
    //    result.ModulusValues[i] = modulusValues[i];
    //    result.AngleValues[i] = angleValues[i];
    //    result.DistanceValues[i] = distanceValues[i];
    //}

    //return ConvertToIntermediate(result, MAX_POINTS);

    // Directly fill the intermediate result
    FillIntermediateResult(externalPoints, modulusValues, angleValues, distanceValues, points, MAX_POINTS, result);
}

// CUDA kernel function
__global__ void ProcessRaysKernel(RayTracingParams rayParams, RenderingParams renderParams,
    int raysPerLine, int totalLines, RayDataTypeIntermediate* results)
{
    int rayCountX = blockIdx.x * blockDim.x + threadIdx.x;
    int rayCountY = blockIdx.y * blockDim.y + threadIdx.y;

    if (rayCountX >= raysPerLine || rayCountY >= totalLines)
        return;


    int index = rayCountY * raysPerLine + rayCountX;
    ProcessRayKernel(rayParams, renderParams, rayCountX, rayCountY, &results[index]);
}
