//#include <vector>
//#include <cmath>
//#include <thread>
//#include <mutex>
//#include "RayProcessing.h"
//#include "TracedRay.h" 
//#include "Clipping.h" 
//
//RayDataTypeIntermediate ProcessRay(RayTracingParams rayParams, RenderingParams renderParams, int rayCountX, int rayCountY) {
//
//    float latitude = rayParams.latitudeStart - rayCountY * rayParams.angularResolution;
//    float longitude = rayParams.longitudeStart - rayCountX * rayParams.angularResolution;
//
//    float latRadians = latitude * DEG_TO_RAD;
//    float longRadians = longitude * DEG_TO_RAD;
//
//    float xFactor = std::cos(latRadians) * std::sin(-longRadians);
//    float yFactor = std::sin(latRadians);
//    float zFactor = std::cos(latRadians) * std::cos(-longRadians);
//
//    float startDistance = rayParams.sphereRadius;
//
//    if (rayParams.useClipping) {
//        float distance = Worlds5::Clipping::CalculateDistance2(latRadians, longRadians, rayParams.clippingAxes, rayParams.clippingOffset);
//        if (distance > startDistance) startDistance = distance;
//    }
//
//    int points = TraceRay(startDistance, rayParams,
//        xFactor, yFactor, zFactor,
//        externalPoints.data(), modulusValues.data(), angleValues.data(), distanceValues.data());
//
//    externalPoints.resize(points);
//    modulusValues.resize(points);
//    angleValues.resize(points);
//    distanceValues.resize(points);
//
//    // Define the TracedRay object
//    TracedRay tracedRay(externalPoints, modulusValues, angleValues, distanceValues, renderParams);
//
//    // Create and return a RayDataType directly
//    TracedRay::RayDataType result = TracedRay::RayDataType{
//        tracedRay.ExternalPoints(),
//        tracedRay.ModulusValues(),
//        tracedRay.AngleValues(),
//        tracedRay.Boundaries(),
//        tracedRay.BoundaryTotal()
//    };
//
//    return ConvertToIntermediate(result);
//}
//
//RayDataTypeIntermediate ConvertToIntermediate(const TracedRay::RayDataType& original) {
//    RayDataTypeIntermediate result;
//    result.ArraySize = original.ExternalPoints.size();
//    result.BoundaryTotal = original.BoundaryTotal;
//
//    result.ExternalPoints = new int[result.ArraySize];
//    result.ModulusValues = new float[result.ArraySize];
//    result.AngleValues = new float[result.ArraySize];
//    result.DistanceValues = new float[result.ArraySize];
//
//    std::copy(original.ExternalPoints.begin(), original.ExternalPoints.end(), result.ExternalPoints);
//    std::copy(original.ModulusValues.begin(), original.ModulusValues.end(), result.ModulusValues);
//    std::copy(original.AngleValues.begin(), original.AngleValues.end(), result.AngleValues);
//    std::copy(original.DistanceValues.begin(), original.DistanceValues.end(), result.DistanceValues);
//
//    return result;
//}
//
///**
//    * Find the distance between the viewpoint and clipping plane
//    * along a vector defined by the lat/long point on the sphere
//    * @param latRadians The latitude in radians of the point on the sphere to trace through
//    * @param longRadians The longitude in radians of the point on the sphere to trace through
//    * @param axisPair The selected axis pair for the clipping plane
//    * @param offset The offset for the clipping plane in the remaining axes
//    * @return The distance value as a float-precision float
//    */
//float CalculateDistance2(float latRadians, float longRadians, AxisPair axisPair, float offset) {
//    // Convert latitude and longitude to a unit direction vector in 3D space
//    Vector3 direction3D(
//        std::cos(latRadians) * std::sin(longRadians),
//        std::sin(latRadians),
//        std::cos(latRadians) * std::cos(longRadians)
//    );
//
//    // Get the 5D coordinates in the fractal space for the sphere centre
//    Vector5 viewpoint5D = ImageToFractalSpace(0, Vector3(0, 0, 0));
//
//    // Get the 5D coordinates in the fractal space for the vector
//    Vector5 direction5D = ImageToFractalSpace(5, direction3D);
//
//    // Determine the intersection parameter t based on the selected axis pair
//    float t = getIntersection2(axisPair, offset, viewpoint5D, direction5D);
//    float distance = std::abs(t * direction5D.Magnitude());
//    return distance;
//}
//
//float getIntersection2(AxisPair axisPair, float offset, const Vector5& viewpoint, const Vector5& direction) {
//    float t = 0;
//    switch (axisPair) {
//    case AxisPair::XY:
//        if (direction.Z != 0) t = (offset - viewpoint.Z) / direction.Z;
//        break;
//    case AxisPair::XZ:
//        if (direction.Y != 0) t = (offset - viewpoint.Y) / direction.Y;
//        break;
//    case AxisPair::XW:
//    case AxisPair::YW:
//    case AxisPair::ZW:
//        if (direction.V != 0) t = (offset - viewpoint.V) / direction.V;
//        break;
//    case AxisPair::XV:
//    case AxisPair::YV:
//    case AxisPair::ZV:
//        if (direction.W != 0) t = (offset - viewpoint.W) / direction.W;
//        break;
//    case AxisPair::YZ:
//    case AxisPair::WV:
//        if (direction.X != 0) t = (offset - viewpoint.X) / direction.X;
//        break;
//    default:
//        throw std::runtime_error("Invalid AxisPair");
//    }
//    return t;
//}