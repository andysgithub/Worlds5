#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

// Forward declarations
struct RGBQUAD;
struct RGBTRIPLE;
class Vector3;
class clsSphere;

// You'll need to define these types and constants in your C++ code
namespace Globals {
    extern const double DEG_TO_RAD;
    // Add other global variables/constants as needed
}

class TracedRay {
public:
    struct RayDataType {
        std::vector<int> ExternalPoints;
        std::vector<float> ModulusValues;
        std::vector<float> AngleValues;
        std::vector<float> DistanceValues;
        int BoundaryTotal;
    };

    TracedRay(const std::vector<int>& externalPoints,
        const std::vector<float>& modulusValues,
        const std::vector<float>& angleValues,
        const std::vector<float>& distanceValues)
        : RayData{ externalPoints, modulusValues, angleValues, distanceValues } {
        RayData.BoundaryTotal = distanceValues.empty() ? 0 : distanceValues.size();
    }

    void SetColour();

    // Getter methods
    int Length() const { return RayData.ExternalPoints.size(); }
    const std::vector<float>& Boundaries() const { return RayData.DistanceValues; }
    const std::vector<float>& AngleValues() const { return RayData.AngleValues; }
    const std::vector<float>& ModulusValues() const { return RayData.ModulusValues; }
    const std::vector<int>& ExternalPoints() const { return RayData.ExternalPoints; }
    int BoundaryTotal() const { return RayData.DistanceValues.empty() ? 0 : RayData.DistanceValues.size(); }
    float Boundary(int index) const { return RayData.DistanceValues[index]; }
    float Angle(int index) const { return RayData.AngleValues[index]; }
    bool IsSurfacePoint(int index) const;

    std::vector<float> xTiltValues;
    std::vector<float> yTiltValues;
    RGBQUAD bmiColors;

private:
    RayDataType RayData;

    void IncreaseRGB(RGBTRIPLE& totalRGB, int i, float Saturation, float Lightness);
    void HSVtoRGB(float h, float s, float v, unsigned char& r, unsigned char& g, unsigned char& b);
};

bool IsPositiveInfinity(float value) {
    return value == std::numeric_limits<float>::infinity();
}

void TracedRay::SetColour() {
    RGBTRIPLE totalRGB = { 0, 0, 0 };

    clsSphere& sphere = Globals::Sphere;

    int activeIndex = sphere.settings.ActiveIndex;

    float totalPoints = sphere.settings.MaxSamples[activeIndex] * sphere.settings.SamplingInterval[activeIndex];
    float startDistance = sphere.settings.SphereRadius;
    float endDistance = startDistance + totalPoints;
    float exposureValue = sphere.settings.ExposureValue[activeIndex];
    float saturation = sphere.settings.Saturation[activeIndex];
    float interiorExposure = sphere.settings.ExposureValue[2];
    float interiorSaturation = sphere.settings.Saturation[2];

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
                    float lightingAngleXY = -sphere.settings.LightingAngle * static_cast<float>(Globals::DEG_TO_RAD);
                    float lightElevationAngle = sphere.settings.LightElevationAngle * static_cast<float>(Globals::DEG_TO_RAD);

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
                    float dotProduct = Vector3::Dot(lightDirection, surfaceNormal);

                    // Ensure the dot product is in the range [0, 1]
                    float tiltValue = std::max(0.0f, dotProduct);

                    float surfaceContrast = sphere.settings.SurfaceContrast / 10.0f;

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
        // Handle error (you might want to use a logging framework instead of MessageBox)
        // std::cerr << "Error tracing ray: " << e.what() << std::endl;
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

void TracedRay::IncreaseRGB(RGBTRIPLE& totalRGB, int i, float Saturation, float Lightness) {
    unsigned char r, g, b;
    clsSphere& sphere = Globals::Sphere;

    float compression = sphere.settings.ColourCompression;
    float offset = sphere.settings.ColourOffset;

    float Hue = (RayData.AngleValues[i] * 57.2957795f * compression) + offset;

    Saturation = std::min(1.0f, Saturation);
    Lightness = std::min(1.0f, Lightness);

    HSVtoRGB(Hue, Saturation, Lightness, r, g, b);

    totalRGB.rgbRed = std::min(255, totalRGB.rgbRed + r);
    totalRGB.rgbGreen = std::min(255, totalRGB.rgbGreen + g);
    totalRGB.rgbBlue = std::min(255, totalRGB.rgbBlue + b);
}

bool TracedRay::IsSurfacePoint(int index) const {
    if (index > 0 && index < BoundaryTotal() && !std::isinf(RayData.DistanceValues[index])) {
        return RayData.ExternalPoints[index - 1] == 1 && RayData.ExternalPoints[index] == 0;
    }
    return false;
}
