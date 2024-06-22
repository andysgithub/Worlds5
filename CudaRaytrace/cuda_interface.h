#pragma once

#include "vector5Double.h"

#ifdef __cplusplus
extern "C" {
#endif

	void launchProcessPointKernel(float* d_Modulus, float* d_Angle, float bailout, vector5Double* d_c, bool* d_result);

	bool launchSamplePointKernel(
		double distance, float* d_Modulus, float* d_Angle, float bailout,
		double xFactor, double yFactor, double zFactor, vector5Double* d_c);

	int launchTraceRayKernel(double startDistance, double increment, double smoothness, double surfaceThickness,
		double XFactor, double YFactor, double ZFactor, float bailout,
		int* externalPoints, float* modulusValues, float* angles, double* distances,
		int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
		int activeIndex);

#ifdef __cplusplus
}
#endif