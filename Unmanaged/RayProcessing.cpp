#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <iostream>
#include <stdio.h>
#include "stdafx.h"
#include "RayProcessing.h"
#include "TracedRay.h" 
#include "Clipping.h" 
#include "Parameters.h"
#include "Vectors.h"

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


        Vector3 rayPoint = Vector3(
            std::cos(latRadians) * std::sin(-longRadians),
            std::sin(latRadians),
            std::cos(latRadians) * std::cos(-longRadians));

        float startDistance = rayParams.sphereRadius;

        if (rayParams.useClipping) {
            float distance = Worlds5::Clipping::CalculateDistance(latRadians, longRadians, rayParams.clippingAxes, rayParams.clippingOffset);
            if (distance > startDistance) startDistance = distance;
        }

        int points = TraceRay(startDistance, rayParams,
            rayPoint,
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

        RayDataTypeIntermediate final = ConvertToIntermediate(result);
        return final;
    }

    RayDataTypeIntermediate ConvertToIntermediate(const TracedRay::RayDataType& original) {
        RayDataTypeIntermediate result;
        result.ArraySize = (int)original.ExternalPoints.size();
        result.BoundaryTotal = original.BoundaryTotal;

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

void ProcessRays(RayTracingParams rayParams, RenderingParams renderParams, int raysPerLine, int totalLines, ProgressCallback callback) {

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
