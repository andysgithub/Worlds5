#pragma once

#include "cuda_interface.h"

#define MAX_POINTS 100

struct RayDataTypeIntermediate {
    int ExternalPoints[MAX_POINTS];
    float ModulusValues[MAX_POINTS];
    float AngleValues[MAX_POINTS];
    float DistanceValues[MAX_POINTS];
    float BoundaryTotal;
    int ArraySize;
};

struct RayDataType {
    int ExternalPoints[MAX_POINTS];
    float ModulusValues[MAX_POINTS];
    float AngleValues[MAX_POINTS];
    float DistanceValues[MAX_POINTS];
    float BoundaryTotal;
    int ArraySize;
};
