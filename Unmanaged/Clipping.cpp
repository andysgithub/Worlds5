#include "Clipping.h"
#include <cmath>
#include <stdexcept>
#include "Vectors.h"
#include "unmanaged.h"

namespace Worlds5 {

    AxisPair Clipping::GetAxes(int axesIndex) {
        return static_cast<AxisPair>(axesIndex);
    }

    /**
        * Find the distance between the viewpoint and clipping plane
        * along a vector defined by the lat/long point on the sphere
        * @param latRadians The latitude in radians of the point on the sphere to trace through
        * @param longRadians The longitude in radians of the point on the sphere to trace through
        * @param axisPair The selected axis pair for the clipping plane
        * @param offset The offset for the clipping plane in the remaining axes
        * @return The distance value as a float-precision float
        */
    float Clipping::CalculateDistance(float latRadians, float longRadians, AxisPair axisPair, float offset) {
        // Convert latitude and longitude to a unit direction vector in 3D space
        Vector3 direction3D(
            std::cos(latRadians) * std::sin(longRadians),
            std::sin(latRadians),
            std::cos(latRadians) * std::cos(longRadians)
        );

        // Get the 5D coordinates in the fractal space for the sphere centre
        Vector5 viewpoint5D = ImageToFractalSpace(0, Vector3(0, 0, 0));

        // Get the 5D coordinates in the fractal space for the vector
        Vector5 direction5D = ImageToFractalSpace(5, direction3D);

        // Determine the intersection parameter t based on the selected axis pair
        float t = getIntersection(axisPair, offset, viewpoint5D, direction5D);
        float distance = std::abs(t * direction5D.Magnitude());
        return distance;
    }

    float Clipping::getIntersection(AxisPair axisPair, float offset, const Vector5& viewpoint, const Vector5& direction) {
        float t = 0;
        switch (axisPair) {
        case AxisPair::XY:
            if (direction.Z != 0) t = (offset - viewpoint.Z) / direction.Z;
            break;
        case AxisPair::XZ:
            if (direction.Y != 0) t = (offset - viewpoint.Y) / direction.Y;
            break;
        case AxisPair::XW:
        case AxisPair::YW:
        case AxisPair::ZW:
            if (direction.V != 0) t = (offset - viewpoint.V) / direction.V;
            break;
        case AxisPair::XV:
        case AxisPair::YV:
        case AxisPair::ZV:
            if (direction.W != 0) t = (offset - viewpoint.W) / direction.W;
            break;
        case AxisPair::YZ:
        case AxisPair::WV:
            if (direction.X != 0) t = (offset - viewpoint.X) / direction.X;
            break;
        default:
            throw std::runtime_error("Invalid AxisPair");
        }
        return t;
    }

} // namespace Worlds5