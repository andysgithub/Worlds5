#pragma once

#include "vector5Double.h"

#ifdef __cplusplus
extern "C" {
#endif

	void launchProcessPointKernel(float* d_Modulus, float* d_Angle, float bailout, vector5Double* d_c, bool* d_result);

#ifdef __cplusplus
}
#endif