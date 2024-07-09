#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include "stdafx.h"
#include "RayProcessing.h"
#include "TracedRay.h" 
#include "Clipping.h" 

class RayProcessing {

public:
    RayProcessing(ProgressCallback callback) : progressCallback(callback) {}

    // Constructor 
    RayProcessing() :
    // Initialiser list
    externalPoints(100), modulusValues(100), angleValues(100), distanceValues(100) {}

    RayDataTypeIntermediate ProcessRay(RayTracingParams rayParams, RenderingParams renderParams, int rayCountX, int rayCountY) {

        float latitude = rayParams.latitudeStart - rayCountY * rayParams.angularResolution;
        float longitude = rayParams.longitudeStart - rayCountX * rayParams.angularResolution;

        float latRadians = latitude * DEG_TO_RAD;
        float longRadians = longitude * DEG_TO_RAD;

        float xFactor = std::cos(latRadians) * std::sin(-longRadians);
        float yFactor = std::sin(latRadians);
        float zFactor = std::cos(latRadians) * std::cos(-longRadians);

        float startDistance = rayParams.sphereRadius;

        if (rayParams.useClipping) {
            float distance = Worlds5::Clipping::CalculateDistance(latRadians, longRadians, rayParams.clippingAxes, rayParams.clippingOffset);
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
        TracedRay::RayDataType result = TracedRay::RayDataType{
            tracedRay.ExternalPoints(),
            tracedRay.ModulusValues(),
            tracedRay.AngleValues(),
            tracedRay.Boundaries(),
            tracedRay.BoundaryTotal()
        };

        return ConvertToIntermediate(result);
    }

    RayDataTypeIntermediate ConvertToIntermediate(const TracedRay::RayDataType& original) {
        RayDataTypeIntermediate result;
        result.ArraySize = original.ExternalPoints.size();
        result.BoundaryTotal = original.BoundaryTotal;

        result.ExternalPoints = new int[result.ArraySize];
        result.ModulusValues = new float[result.ArraySize];
        result.AngleValues = new float[result.ArraySize];
        result.DistanceValues = new float[result.ArraySize];

        std::copy(original.ExternalPoints.begin(), original.ExternalPoints.end(), result.ExternalPoints);
        std::copy(original.ModulusValues.begin(), original.ModulusValues.end(), result.ModulusValues);
        std::copy(original.AngleValues.begin(), original.AngleValues.end(), result.AngleValues);
        std::copy(original.DistanceValues.begin(), original.DistanceValues.end(), result.DistanceValues);

        return result;
    }

private:
    std::vector<int> externalPoints;
    std::vector<float> modulusValues;
    std::vector<float> angleValues;
    std::vector<float> distanceValues;

    ProgressCallback progressCallback;
};

// Example of how to use std::thread for parallel processing in C++
EXPORT void __stdcall ProcessRays(RayTracingParams rayParams, RenderingParams renderParams, int raysPerLine, int totalLines, ProgressCallback callback) {
    std::vector<std::vector<RayProcessing>> rayProc(raysPerLine, std::vector<RayProcessing>(totalLines));

    std::vector<std::thread> threads;
    std::mutex mtx;

    for (int rayCountY = 0; rayCountY < totalLines; ++rayCountY) {
        threads.emplace_back([&, rayCountY]() {
            for (int rayCountX = 0; rayCountX < raysPerLine; ++rayCountX) {
                try {
                    RayDataTypeIntermediate rayData = rayProc[rayCountX][rayCountY].ProcessRay(rayParams, renderParams, rayCountX, rayCountY);

                    if (callback) {
                        std::lock_guard<std::mutex> lock(mtx);
                        callback(rayCountX, rayCountY, &rayData);
                    }
                }
                catch (...) {
                    // Handle exceptions
                }
            }
            });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void FreeIntermediateData(RayDataTypeIntermediate& data) {
    delete[] data.ExternalPoints;
    delete[] data.ModulusValues;
    delete[] data.AngleValues;
    delete[] data.DistanceValues;
}

EXPORT void __stdcall FreeRayData(RayDataTypeIntermediate * data) {
    FreeIntermediateData(*data);
}
