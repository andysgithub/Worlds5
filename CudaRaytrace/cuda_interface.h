#pragma once

#include "vector5Double.h"

struct RayTracingParams {
    double startDistance;
    double increment;
    double smoothness;
    double surfaceThickness;
    float bailout;
    int rayPoints;
    int maxSamples;
    double boundaryInterval;
    int binarySearchSteps;
    int activeIndex;
};

// Declare the constant symbol
//extern __constant__ RayTracingParams d_params;

#ifdef __cplusplus
extern "C" {
#endif

	cudaError_t InitializeGPUKernel(const RayTracingParams* params);

    cudaError_t InitializeTransformMatrix(const double* positionMatrix);
    cudaError_t VerifyTransformMatrix(double* output);

	int launchTraceRayKernel(double XFactor, double YFactor, double ZFactor, int rayPoints,
		int* externalPoints, float* modulusValues, float* angles, double* distances);

#ifdef __cplusplus
}
#endif