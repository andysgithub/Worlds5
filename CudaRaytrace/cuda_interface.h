#pragma once

#include "vector5Double.h"

#ifdef __cplusplus
extern "C" {
#endif

	void InitializeGPUKernel(const RayTracingParams* params);

	int launchTraceRayKernel(double XFactor, double YFactor, double ZFactor, int rayPoints,
		int* externalPoints, float* modulusValues, float* angles, double* distances);

#ifdef __cplusplus
}
#endif