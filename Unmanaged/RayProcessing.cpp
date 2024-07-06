#include <vector>
#include <cmath>
#include <thread>
#include <mutex>

// Forward declaration of clsSphere (you'll need to define this class in C++)
class clsSphere;

class RayProcessing {

    typedef void (*ProgressCallback)(int rayCount, int rowCount);

public:
    RayProcessing(ProgressCallback callback) : progressCallback(callback) {}

    struct RayTracingParams {
        int activeIndex;
        float angularResolution;
        float bailout;
        int binarySearchSteps;
        float boundaryInterval;
        bool cudaMode;
        int maxSamples;
        int rayPoints;
        float samplingInterval;
        float sphereRadius;
        float surfaceSmoothing;
        float surfaceThickness;
    };

    RayProcessing()
        : externalPoints(100), modulusValues(100), angleValues(100), distanceValues(100) {}

    void ProcessRay(clsSphere& sphere, int rayCountX, int rayCountY) {
        // You'll need to implement clsSphere and its Settings in C++
        auto& settings = sphere.settings;
        float latitude = settings.LatitudeStart - rayCountY * settings.AngularResolution;
        float longitude = settings.LongitudeStart - rayCountX * settings.AngularResolution;
        int i = settings.ActiveIndex;
        int rayPoints = static_cast<int>(settings.MaxSamples[i] * settings.SamplingInterval[i]);

        constexpr float DEG_TO_RAD = 3.14159265358979323846f / 180.0f;
        float latRadians = latitude * DEG_TO_RAD;
        float longRadians = longitude * DEG_TO_RAD;

        float xFactor = std::cos(latRadians) * std::sin(-longRadians);
        float yFactor = std::sin(latRadians);
        float zFactor = std::cos(latRadians) * std::cos(-longRadians);

        float startDistance = settings.SphereRadius;

        if (settings.UseClipping) {
            // Implement Clipping::CalculateDistance in C++
            float distance = Clipping::CalculateDistance(latRadians, longRadians, settings.ClippingAxes, settings.ClippingOffset);
            if (distance > startDistance) startDistance = distance;
        }

        int points = TraceRay(startDistance, settings.SamplingInterval[i], settings.SurfaceSmoothing, settings.SurfaceThickness,
            xFactor, yFactor, zFactor, settings.Bailout,
            externalPoints.data(), modulusValues.data(), angleValues.data(), distanceValues.data(),
            rayPoints, settings.MaxSamples[i], settings.BoundaryInterval, settings.BinarySearchSteps[i],
            i, settings.CudaMode);

        externalPoints.resize(points);
        modulusValues.resize(points);
        angleValues.resize(points);
        distanceValues.resize(points);

        // Implement TracedRay in C++
        TracedRay tracedRay(externalPoints, modulusValues, angleValues, distanceValues);

        // Update sphere's RayMap (you'll need to implement this in C++)
        sphere.UpdateRayMap(rayCountX, rayCountY, tracedRay.RayData);
    }

private:
    std::vector<int> externalPoints;
    std::vector<float> modulusValues;
    std::vector<float> angleValues;
    std::vector<float> distanceValues;

    ProgressCallback progressCallback;

    // Declare TraceRay as a member function or keep it as an external function
    static int TraceRay(float startDistance, float samplingInterval, float surfaceSmoothing, float surfaceThickness,
        float xFactor, float yFactor, float zFactor, float bailout,
        int* externalsArray, float* valuesArray, float* anglesArray, float* distancesArray,
        int rayPoints, int maxSamples, float boundaryInterval, int binarySearchSteps,
        int activeIndex, bool cudaMode);
};
