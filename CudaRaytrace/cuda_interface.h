#pragma once

#include "vector5Single.h"

struct RayTracingParams {
    int activeIndex;
    float angularResolution;
    float bailout;
    int binarySearchSteps;
    float boundaryInterval;
    bool cudaMode;
    int maxSamples;
    int rayPoints;
    float samplingInterval;
    float sphereRadius;
    float surfaceSmoothing;
    float surfaceThickness;
};

// Declare the constant symbol
//extern __constant__ RayTracingParams d_params;

#ifdef __cplusplus
extern "C" {
#endif

	cudaError_t InitializeGPUKernel(const RayTracingParams* params);

    cudaError_t InitializeTransformMatrix(const float* positionMatrix);

	int launchTraceRayKernel(float XFactor, float YFactor, float ZFactor, int rayPoints,
		int* externalPoints, float* modulusValues, float* angles, float* distances);

#ifdef __cplusplus
}
#endif