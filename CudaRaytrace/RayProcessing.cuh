#pragma once

#include <cuda_runtime.h>
#include "cuda_interface.h"

struct RayDataTypeIntermediate {
    int ExternalPoints[MAX_POINTS];
    float ModulusValues[MAX_POINTS];
    float AngleValues[MAX_POINTS];
    float DistanceValues[MAX_POINTS];
    int BoundaryTotal;
    int ArraySize;
};

struct RayDataType {
    int ExternalPoints[MAX_POINTS];
    float ModulusValues[MAX_POINTS];
    float AngleValues[MAX_POINTS];
    float DistanceValues[MAX_POINTS];
    int BoundaryTotal;
    int ArraySize;
};

__global__ void ProcessRaysKernel(RayTracingParams rayParams, RenderingParams renderParams,
    int raysPerLine, int totalLines, RayDataTypeIntermediate* results);