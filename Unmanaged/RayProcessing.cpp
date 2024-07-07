#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include "stdafx.h"
#include "unmanaged.h"
#include "TracedRay.h" 

class RayProcessing {

    typedef void (*ProgressCallback)(int rayCount, int rowCount);

public:
    RayProcessing(ProgressCallback callback) : progressCallback(callback) {}

    // Constructor 
    RayProcessing() :
    // Initialiser list
    externalPoints(100), modulusValues(100), angleValues(100), distanceValues(100) {}

    TracedRay::RayDataType ProcessRay(RayTracingParams rayParams, RenderingParams renderParams, int rayCountX, int rayCountY) {

        float latitude = rayParams.latitudeStart - rayCountY * rayParams.angularResolution;
        float longitude = rayParams.longitudeStart - rayCountX * rayParams.angularResolution;
        int rayPoints = static_cast<int>(rayParams.maxSamples * rayParams.samplingInterval);

        constexpr float DEG_TO_RAD = 3.14159265358979323846f / 180.0f;
        float latRadians = latitude * DEG_TO_RAD;
        float longRadians = longitude * DEG_TO_RAD;

        float xFactor = std::cos(latRadians) * std::sin(-longRadians);
        float yFactor = std::sin(latRadians);
        float zFactor = std::cos(latRadians) * std::cos(-longRadians);

        float startDistance = rayParams.sphereRadius;

        if (rayParams.useClipping) {
            // Implement Clipping::CalculateDistance in C++
            float distance = Clipping::CalculateDistance(latRadians, longRadians, rayParams.clippingAxes, rayParams.clippingOffset);
            if (distance > startDistance) startDistance = distance;
        }

        int points = TraceRay(startDistance, rayParams,
            xFactor, yFactor, zFactor,
            externalPoints.data(), modulusValues.data(), angleValues.data(), distanceValues.data());

        externalPoints.resize(points);
        modulusValues.resize(points);
        angleValues.resize(points);
        distanceValues.resize(points);

        // Define the TracedRay object
        TracedRay tracedRay(externalPoints, modulusValues, angleValues, distanceValues, renderParams);

        // Create and return a RayDataType directly
        return TracedRay::RayDataType{
            tracedRay.ExternalPoints(),
            tracedRay.ModulusValues(),
            tracedRay.AngleValues(),
            tracedRay.Boundaries(),
            tracedRay.BoundaryTotal()
        };
    }

private:
    std::vector<int> externalPoints;
    std::vector<float> modulusValues;
    std::vector<float> angleValues;
    std::vector<float> distanceValues;

    ProgressCallback progressCallback;
};
