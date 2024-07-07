#pragma once

#include "stdafx.h"
#include <vector>
#include <windows.h>
#include "cuda_interface.h"

class TracedRay {
public:
    struct RayDataType {
        std::vector<int> ExternalPoints;
        std::vector<float> ModulusValues;
        std::vector<float> AngleValues;
        std::vector<float> DistanceValues;
        int BoundaryTotal;
    };

    //TracedRay(
    //    const std::vector<int>& externalPoints,
    //    const std::vector<float>& modulusValues,
    //    const std::vector<float>& angleValues,
    //    const std::vector<float>& distanceValues,
    //    RenderingParams renderParams);

    void SetColour();

    // Getter methods
    int Length() const;
    const std::vector<float>& Boundaries() const;
    const std::vector<float>& AngleValues() const;
    const std::vector<float>& ModulusValues() const;
    const std::vector<int>& ExternalPoints() const;
    int BoundaryTotal() const;
    float Boundary(int index) const;
    float Angle(int index) const;
    bool IsSurfacePoint(int index) const;

    std::vector<float> xTiltValues;
    std::vector<float> yTiltValues;
    RGBQUAD bmiColors;

private:
    RayDataType RayData;
    RenderingParams m_renderParams;

    void IncreaseRGB(RGBTRIPLE& totalRGB, int i, float Saturation, float Lightness);
    void HSVtoRGB(float h, float s, float v, unsigned char& r, unsigned char& g, unsigned char& b);
};

bool IsPositiveInfinity(float value);
