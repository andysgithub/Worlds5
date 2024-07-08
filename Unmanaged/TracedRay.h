#pragma once

#ifndef TRACED_RAY_H
#define TRACED_RAY_H

#define NOMINMAX

#include <vector>
#include <cmath>
#include "vectors.h"
#include "cuda_interface.h"
#include "Colour.h"

// Define RGB_QUAD and RGB_TRIPLE if they're not already defined elsewhere

struct RGB_QUAD {
    unsigned char rgbBlue;
    unsigned char rgbGreen;
    unsigned char rgbRed;
    unsigned char rgbReserved;
};

struct RGB_TRIPLE {
    unsigned char rgbBlue;
    unsigned char rgbGreen;
    unsigned char rgbRed;
};

class TracedRay {
public:

    struct RayDataType {
        std::vector<int> ExternalPoints;
        std::vector<float> ModulusValues;
        std::vector<float> AngleValues;
        std::vector<float> DistanceValues;
        int BoundaryTotal;
    };

    TracedRay(
        const std::vector<int>& externalPoints,
        const std::vector<float>& modulusValues,
        const std::vector<float>& angleValues,
        const std::vector<float>& distanceValues,
        RenderingParams renderParams);

    void SetColour();

    // Getter methods
    int Length() const { 
        return (int)RayData.ExternalPoints.size(); 
    }
    const std::vector<float>& Boundaries() const { 
        return RayData.DistanceValues; 
    }
    const std::vector<float>& AngleValues() const { 
        return RayData.AngleValues; 
    }
    const std::vector<float>& ModulusValues() const { 
        return RayData.ModulusValues; 
    }
    const std::vector<int>& ExternalPoints() const { 
        return RayData.ExternalPoints; 
    }
    int BoundaryTotal() const { 
        return RayData.DistanceValues.empty() ? 0 : (int)RayData.DistanceValues.size(); 
    }
    float Boundary(int index) const { 
        return RayData.DistanceValues[index]; 
    }
    float Angle(int index) const { 
        return RayData.AngleValues[index]; 
    }
    bool IsSurfacePoint(int index) const;

    std::vector<float> xTiltValues;
    std::vector<float> yTiltValues;
    RGB_QUAD bmiColors;

private:
    RayDataType RayData;
    RenderingParams m_renderParams;

    void IncreaseRGB(RGB_TRIPLE& totalRGB, int i, float Saturation, float Lightness);
};

bool IsPositiveInfinity(float value);

#endif