#define NOMINMAX

#include "TracedRay.h"
#include "Colour.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <Windows.h>
#include "Vectors.cuh"
#include "cuda_interface.h"

// Constructor taking 4 references to vector objects
TracedRay::TracedRay(
    const std::vector<int>& externalPoints,
    const std::vector<float>& modulusValues,
    const std::vector<float>& angleValues,
    const std::vector<float>& distanceValues,
    RenderingParams renderParams) 
    :
    // Initialiser list - initialise RayData object from the vectors
    RayData{ externalPoints, modulusValues, angleValues, distanceValues, 0 }, m_renderParams(renderParams)
    {
        // Constructor body - set the BoundaryTotal member
        RayData.BoundaryTotal = distanceValues.empty() ? 0 : (int)distanceValues.size();
    }


bool IsPositiveInfinity(float value) {
    return value == std::numeric_limits<float>::infinity();
}

void TracedRay::IncreaseRGB(RGB_TRIPLE& totalRGB, int i, float Saturation, float Lightness) {
    unsigned char r, g, b;

    float compression = m_renderParams.colourCompression;
    float offset = m_renderParams.colourOffset;

    float Hue = (RayData.AngleValues[i] * 57.2957795f * compression) + offset;

    Saturation = std::min(1.0f, Saturation);
    Lightness = std::min(1.0f, Lightness);

    HSVtoRGB(Hue, Saturation, Lightness, r, g, b);

    totalRGB.rgbRed = static_cast<unsigned char>(std::min(255, totalRGB.rgbRed + r));
    totalRGB.rgbGreen = static_cast<unsigned char>(std::min(255, totalRGB.rgbGreen + g));
    totalRGB.rgbBlue = static_cast<unsigned char>(std::min(255, totalRGB.rgbBlue + b));
}

bool TracedRay::IsSurfacePoint(int index) const {
    if (index > 0 && index < BoundaryTotal() && !std::isinf(RayData.DistanceValues[index])) {
        return RayData.ExternalPoints[index - 1] == 1 && RayData.ExternalPoints[index] == 0;
    }
    return false;
}

void TracedRay::SetColour() {
    RGB_TRIPLE totalRGB = { 0, 0, 0 };

    int activeIndex = m_renderParams.activeIndex;
    float startDistance = m_renderParams.startDistance;
    float endDistance = m_renderParams.endDistance;
    float exposureValue = m_renderParams.exposureValue;
    float saturation = m_renderParams.saturation;
    float interiorExposure = m_renderParams.interiorExposure;
    float interiorSaturation = m_renderParams.interiorSaturation;

    try {
        for (int i = 0; i < RayData.ModulusValues.size() - 1; i++) {
            if (RayData.DistanceValues[i] < startDistance)
                continue;

            if (std::abs(RayData.ModulusValues[i]) < 10) {
                if (activeIndex == 0 && IsSurfacePoint(i) && !xTiltValues.empty() && !yTiltValues.empty()) {

                    ///// Set colour for surface point /////

                    if (IsPositiveInfinity(RayData.DistanceValues[i]) || RayData.DistanceValues[i] > endDistance)
                        break;

                    // Light position parameters
                    float lightingAngleXY = -m_renderParams.lightingAngle * DEG_TO_RAD;
                    float lightElevationAngle = m_renderParams.lightElevationAngle * DEG_TO_RAD;

                    // Calculate light direction
                    Vector3 lightDirection(
                        static_cast<float>(std::cos(lightingAngleXY) * std::cos(lightElevationAngle)),
                        static_cast<float>(std::sin(lightingAngleXY) * std::cos(lightElevationAngle)),
                        static_cast<float>(std::sin(lightElevationAngle))
                    );
                    lightDirection = lightDirection.Normalize();

                    // Get tilt values
                    float xTilt = !xTiltValues.empty() ? xTiltValues[i] : 0.0f;
                    float yTilt = !yTiltValues.empty() ? yTiltValues[i] : 0.0f;

                    // Calculate the surface normal
                    Vector3 surfaceNormal(
                        static_cast<float>(-std::sin(xTilt)),
                        static_cast<float>(-std::sin(yTilt)),
                        static_cast<float>(std::sqrt(1 - std::sin(xTilt) * std::sin(xTilt) - std::sin(yTilt) * std::sin(yTilt)))
                    );
                    surfaceNormal = surfaceNormal.Normalize();

                    // Calculate the dot product
                    float dotProduct = lightDirection.dot(surfaceNormal);

                    // Ensure the dot product is in the range [0, 1]
                    float tiltValue = std::max(0.0f, dotProduct);

                    float surfaceContrast = m_renderParams.surfaceContrast / 10.0f;

                    // Apply contrast
                    float contrastValue = (tiltValue - 0.5f) * (1 + surfaceContrast) + 0.5f;
                    contrastValue = std::max(0.0f, std::min(1.0f, contrastValue));

                    // Calculate final lightness
                    float lightness = contrastValue * exposureValue / 10.0f;
                    lightness = std::max(0.0f, std::min(1.0f, lightness));

                    // Modify the exposure according to the position of the point between the start and end distances
                    // float range = endDistance - startDistance;
                    // float exposureFactor = (RayData.DistanceValues[i] - startDistance) / range;
                    // exposureFactor = std::min(1.0f, exposureFactor);

                    // S & V set by exposure value
                    float Lightness = lightness; // * (1 - exposureFactor);
                    float Saturation = Lightness * saturation / 10.0f;

                    IncreaseRGB(totalRGB, i, Saturation, Lightness);

                }
                else if (activeIndex == 1) {
                    ///// Set colour for volume point /////

                    if (IsPositiveInfinity(RayData.DistanceValues[i + 1]) || RayData.DistanceValues[i + 1] > endDistance)
                        break;

                    // Get distance between points
                    float distance = RayData.DistanceValues[i + 1] - RayData.DistanceValues[i];

                    // S & V set by distance * exposure value
                    float Lightness = distance * exposureValue / 10.0f;
                    float Saturation = Lightness * saturation;

                    IncreaseRGB(totalRGB, i, Saturation, Lightness);

                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error tracing ray: " << e.what() << std::endl;
    }

    // Color the inside of the set if visible
    if (RayData.ModulusValues.size() == 2 && RayData.ModulusValues[0] < 2 && RayData.ModulusValues[1] == 0) {
        float Lightness = interiorExposure / 10;
        float Saturation = Lightness * interiorSaturation / 10;
        IncreaseRGB(totalRGB, 0, Saturation, Lightness);
    }

    bmiColors.rgbRed = static_cast<unsigned char>(totalRGB.rgbRed);
    bmiColors.rgbGreen = static_cast<unsigned char>(totalRGB.rgbGreen);
    bmiColors.rgbBlue = static_cast<unsigned char>(totalRGB.rgbBlue);
}
