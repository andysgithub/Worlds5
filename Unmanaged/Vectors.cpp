#include "stdafx.h"
#include <math.h>
#include "Vectors.cuh"

// Return angle between lines defined by vectors BA and BC
float vectorAngle(Vector5 A, Vector5 B, Vector5 C)
{
    Vector5 v1, v2;
    float dotProduct = 0;

    // Vector v1 = B - A 
    v1.m[0] = B.m[0] - A.m[0];
    v1.m[1] = B.m[1] - A.m[1];
    v1.m[2] = B.m[2] - A.m[2];
    v1.m[3] = B.m[3] - A.m[3];
    v1.m[4] = B.m[4] - A.m[4];

    float modulus = static_cast<float>(sqrt(v1.m[0]*v1.m[0] + v1.m[1]*v1.m[1] +
                          v1.m[2]*v1.m[2] + v1.m[3]*v1.m[3] + 
                          v1.m[4]*v1.m[4]));

    if (modulus != 0)
    {
        float factor = 1 / modulus;

        // Normalise v1 by dividing by mod(v1)
        v1.m[0] = v1.m[0] * factor;
        v1.m[1] = v1.m[1] * factor;
        v1.m[2] = v1.m[2] * factor;
        v1.m[3] = v1.m[3] * factor;
        v1.m[4] = v1.m[4] * factor;

        // Vector v2 = B - C 
        v2.m[0] = B.m[0] - C.m[0];
        v2.m[1] = B.m[1] - C.m[1];
        v2.m[2] = B.m[2] - C.m[2];
        v2.m[3] = B.m[3] - C.m[3];
        v2.m[4] = B.m[4] - C.m[4];

        modulus = static_cast<float>(sqrt(v2.m[0]*v2.m[0] + v2.m[1]*v2.m[1] +
                       v2.m[2]*v2.m[2] + v2.m[3]*v2.m[3] + 
                       v2.m[4]*v2.m[4]));

        if (modulus != 0)
        {
            factor = 1 / modulus;

            // Normalise v2 by dividing by mod(v2)
            v2.m[0] = v2.m[0] * factor;
            v2.m[1] = v2.m[1] * factor;
            v2.m[2] = v2.m[2] * factor;
            v2.m[3] = v2.m[3] * factor;
            v2.m[4] = v2.m[4] * factor;

            // Calculate dot product of v1 and v2
            dotProduct = v1.m[0]*v2.m[0] + v1.m[1]*v2.m[1] +
                         v1.m[2]*v2.m[2] + v1.m[3]*v2.m[3] + 
                         v1.m[4]*v2.m[4];
        }
    }

    if (dotProduct > 1)
    {
        dotProduct = 1;
    }
    else if (dotProduct < -1)
    {
        dotProduct = -1;
    }

    // Return the angle in radians
    return static_cast<float>(acos(dotProduct));
}
