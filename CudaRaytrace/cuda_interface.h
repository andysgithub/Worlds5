#pragma once

#include <cuda_runtime.h>
#include <vector>

enum class AxisPair
{
    XY,
    XZ,
    XW,
    XV,
    YZ,
    YW,
    YV,
    ZW,
    ZV,
    WV
};

struct __align__(8) RayTracingParams {
    int activeIndex;
    float angularResolution;
    float bailout;
    int binarySearchSteps;
    float boundaryInterval;
    AxisPair clippingAxes;
    float clippingOffset;
    bool cudaMode;
    float latitudeStart;
    float longitudeStart;
    int maxSamples;
    float samplingInterval;
    float sphereRadius;
    float surfaceSmoothing;
    float surfaceThickness;
    bool useClipping;
};

struct __align__(8) RenderingParams {
    int activeIndex;
    float startDistance;
    float endDistance;
    float exposureValue;
    float interiorExposure;
    float saturation;
    float interiorSaturation;
    float lightingAngle;
    float lightElevationAngle;
    float surfaceContrast;
    float colourCompression;
    float colourOffset;
};

struct RayDataType
{
    std::vector<int> ExternalPoints;
    std::vector<float> ModulusValues;
    std::vector<float> AngleValues;
    std::vector<float> DistanceValues;
    int BoundaryTotal;
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
