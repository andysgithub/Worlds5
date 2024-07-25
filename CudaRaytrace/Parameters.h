#pragma once

#include <vector>

#define MAX_POINTS 100

#define DEG_TO_RAD 0.0174532925F

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

struct RayTracingParams {
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

struct RenderingParams {
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

// Forward declarations
struct RayTracingParams;
struct RenderingParams;
struct RayDataTypeIntermediate;

// Define the callback type after RayDataTypeIntermediate is declared
typedef void(__stdcall* ProgressCallback)(int rayCount, int rowCount, RayDataTypeIntermediate* rayData);
