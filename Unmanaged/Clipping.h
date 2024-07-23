#pragma once

#include "cuda_interface.h"
#include "Vectors.cuh"

namespace Worlds5 {

    class Clipping {
    public:
        static AxisPair GetAxes(int axesIndex);

        static float CalculateDistance(float latRadians, float longRadians, AxisPair axisPair, float offset);

    private:
        static float getIntersection(AxisPair axisPair, float offset, const Vector5& viewpoint, const Vector5& direction);
    };

} // namespace Worlds5