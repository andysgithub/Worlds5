#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdio.h>
#include "inline.cuh"
#include "cuda_interface.h"
#include "RayTracer.cuh"

__constant__ float cudaTrans[6][DimTotal];

namespace RayTracer {

    __device__ int TraceRay(float startDistance, const RayTracingParams* __restrict__ rayParams,
        float xFactor, float yFactor, float zFactor, int rayPoints,
        int* __restrict__ externalPoints, float* __restrict__ modulusValues,
        float* __restrict__ angles, float* __restrict__ distances) {

        const vector5Single c = { 0, 0, 0, 0, 0 };
        float Modulus, Angle, currentDistance = startDistance;
        int recordedPoints = 0, sampleCount = 0;
        bool externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayParams->bailout, xFactor, yFactor, zFactor, c);

        if (recordedPoints < rayPoints) {
            externalPoints[recordedPoints] = externalPoint ? 1 : 0;
            modulusValues[recordedPoints] = Modulus;
            angles[recordedPoints] = Angle;
            distances[recordedPoints] = currentDistance;
            recordedPoints++;
        }

        if (rayParams->activeIndex == 0) {
            bool previousPointExternal = true;
            while (recordedPoints < rayPoints && sampleCount < rayParams->maxSamples) {
                currentDistance += rayParams->samplingInterval;
                sampleCount++;

                externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayParams->bailout, xFactor, yFactor, zFactor, c);

                bool shouldRecord = !externalPoint && previousPointExternal;
                float sampleDistance = currentDistance;

                if (shouldRecord) {
                    //sampleDistance = FindSurface(
                    //    rayParams->samplingInterval, rayParams->surfaceSmoothing, rayParams->binarySearchSteps,
                    //    currentDistance, xFactor, yFactor, zFactor, rayParams->bailout);
                    //bool foundGap = gapFound(sampleDistance, rayParams->surfaceThickness, xFactor, yFactor, zFactor, rayParams->bailout, c);

                    //if (rayParams->surfaceThickness > 0 && foundGap) {
                    //    previousPointExternal = true;
                    //    continue;
                    //}
                    //externalPoint = SamplePoint(sampleDistance, &Modulus, &Angle, rayParams->bailout, xFactor, yFactor, zFactor, c);

                    externalPoints[recordedPoints] = externalPoint ? 1 : 0;
                    modulusValues[recordedPoints] = Modulus;
                    angles[recordedPoints] = Angle;
                    distances[recordedPoints] = sampleDistance;
                    recordedPoints++;
                }

                previousPointExternal = externalPoint;
            }

            if (rayParams->activeIndex == 1) {
                while (recordedPoints < rayPoints && sampleCount < rayParams->maxSamples) {
                    currentDistance += rayParams->samplingInterval;
                    sampleCount++;

                    externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayParams->bailout, xFactor, yFactor, zFactor, c);

                    ///// Set value for external point /////

                    float angleChange = fabs(Angle - angles[recordedPoints - 1]);

                    // If orbit value is sufficiently different from the last recorded sample
                    if (angleChange > rayParams->boundaryInterval) {
                        // Perform binary search between this and the recorded point, to determine boundary position
                        float sampleDistance = FindBoundary(rayParams->samplingInterval, rayParams->binarySearchSteps, currentDistance, angles[recordedPoints - 1],
                            rayParams->boundaryInterval, &externalPoint, &Modulus, &Angle,
                            xFactor, yFactor, zFactor, rayParams->bailout);

                        // Save this point value in the ray collection
                        externalPoints[recordedPoints] = externalPoint ? 1 : 0;
                        modulusValues[recordedPoints] = Modulus;
                        angles[recordedPoints] = Angle;
                        distances[recordedPoints] = sampleDistance;
                        recordedPoints++;
                    }
                }
            }
        }

        if (recordedPoints < rayPoints) {
            distances[recordedPoints] = CUDART_INF_F;
        }
        return recordedPoints + 1;
    }

    __device__ float FindSurface(
        float samplingInterval, float surfaceSmoothing, int binarySearchSteps, float currentDistance,
        float xFactor, float yFactor, float zFactor, float bailout) {

        float stepFactor = surfaceSmoothing / 10;
        float stepSize = -samplingInterval * stepFactor;
        float sampleDistance = currentDistance;
        const vector5Single c = { 0, 0, 0, 0, 0 };

        for (int i = 0; i < binarySearchSteps; i++) {
            sampleDistance += stepSize;

            if (!SamplePoint(sampleDistance, bailout, xFactor, yFactor, zFactor, c)) {
                stepSize = -fabs(stepSize) * stepFactor;
            }
            else {
                stepSize = fabs(stepSize) * stepFactor;
            }
        }
        return sampleDistance;
    }

    __device__ float FindBoundary(float samplingInterval, int binarySearchSteps, float currentDistance, float previousAngle,
        float boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
        float xFactor, float yFactor, float zFactor, float bailout) {
        float stepSize = -samplingInterval / 2;
        float sampleDistance = currentDistance;
        const vector5Single c = { 0, 0, 0, 0, 0 };

        for (int i = 0; i < binarySearchSteps; i++) {
            sampleDistance += stepSize;
            *externalPoint = SamplePoint(sampleDistance, Modulus, Angle, bailout, xFactor, yFactor, zFactor, c);

            const float angleChange = fabs(*Angle - previousAngle);

            if (angleChange > boundaryInterval) {
                stepSize = -fabs(stepSize) / 2;
            }
            else {
                stepSize = fabs(stepSize) / 2;
            }
        }
        return sampleDistance;
    }

    __device__ bool SamplePoint(float distance, float* Modulus, float* Angle, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c) {
        // Determine the x,y,z coord for this point
        const float XPos = distance * xFactor;
        const float YPos = distance * yFactor;
        const float ZPos = distance * zFactor;

        // Transform 3D point x,y,z into nD fractal space at point c[]
        VectorTrans(XPos, YPos, ZPos, &c);

        // Determine orbit value for this point
        return ProcessPoint(Modulus, Angle, bailout, c);
    }

    __device__ bool SamplePoint(float distance, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c) {
        // Determine the x,y,z coord for this point
        const float XPos = distance * xFactor;
        const float YPos = distance * yFactor;
        const float ZPos = distance * zFactor;

        // Transform 3D point x,y,z into nD fractal space at point c[]
        VectorTrans(XPos, YPos, ZPos, &c);

        // Determine orbit value for this point
        return ExternalPoint(c, bailout);
    }

    // Determine whether nD point c[] in within the set
    // Returns true if point is external to the set
    __device__ bool ExternalPoint(vector5Single c, float bailout)
    {
        const long MaxCount = (long)(1000);		        // Iteration count for external points
        vector5Single z;										// Temporary 5-D vector
        vector5Single diff;										// Temporary 5-D vector for orbit size
        float ModulusTotal = 0;
        float ModVal = 0;
        long count;

        z.coords[DimTotal - 2] = 0;
        z.coords[DimTotal - 1] = 0;

        v_mov(c.coords, z.coords);        // z = c

        for (count = 0; count < MaxCount; count++)
        {
            v_mandel(z.coords, c.coords);                   //    z = z*z + c

            // Determine modulus for this point in orbit
            v_subm(c.coords, z.coords, diff.coords);        // Current orbit size = mod(z - c)
            ModVal = v_mod(diff.coords);

            // Accumulate modulus value
            ModulusTotal += ModVal;

            // Stop accumulating values when modulus exceeds bailout value
            if (ModVal > bailout * bailout)
            {
                count++;
                break;
            }
        }

        // Return true if this point is external to the set
        return (count < MaxCount);
    }

    __device__ bool ProcessPoint(float* Modulus, float* Angle, float bailout, vector5Single c) {
        constexpr float PI = 3.1415926536f;

        constexpr int MaxCount = 100;
        vector5Single z = { 0 };
        vector5Single diff;
        float ModulusTotal = 0;
        float ModVal = 0;
        float AngleTotal = PI;
        int count;

        z.coords[DimTotal - 2] = 0;
        z.coords[DimTotal - 1] = 0;

        v_mov(c.coords, z.coords);
        vector5Single vectorSet[3];
        v_mov(z.coords, vectorSet[1].coords);

        float bailout_squared = bailout * bailout;

        for (count = 0; count < MaxCount; count++) {
            v_mandel(z.coords, c.coords);
            v_mov(z.coords, vectorSet[2].coords);

            if (count > 0 && count < 10) {
                AngleTotal += vectorAngle(vectorSet[0], vectorSet[1], vectorSet[2]);
            }

            v_subm(c.coords, z.coords, diff.coords);
            ModVal = v_mod(diff.coords);

            ModulusTotal += ModVal;

            if (ModVal > bailout_squared) {
                count++;
                break;
            }

            v_mov(vectorSet[1].coords, vectorSet[0].coords);
            v_mov(vectorSet[2].coords, vectorSet[1].coords);
        }

        *Modulus = ModulusTotal / count;
        *Angle = AngleTotal / (count > 10 ? 10 : count + 1);

        return (count < MaxCount);
    }

    __device__ bool gapFound(float currentDistance, float surfaceThickness, float xFactor, float yFactor, float zFactor, float bailout, vector5Single c) {
        float testDistance;

        for (int factor = 1; factor <= 4; factor++) {
            testDistance = currentDistance + surfaceThickness * factor / 4;

            if (SamplePoint(testDistance, bailout, xFactor, yFactor, zFactor, c)) {
                return true;
            }
        }
        return false;
    }

    __device__ void VectorTrans(float x, float y, float z, vector5Single* c) {
        for (int col = 0; col < DimTotal; col++) {
            (*c).coords[col] =
                cudaTrans[0][col] * x +
                cudaTrans[1][col] * y +
                cudaTrans[2][col] * z +
                cudaTrans[5][col];
        }
    }

    __device__ float vectorAngle(const vector5Single& A, const vector5Single& B, const vector5Single& C) {
        vector5Single v1, v2;
        float dotProduct = 0.0;

        // Vector v1 = B - A 
        v1.coords[0] = B.coords[0] - A.coords[0];
        v1.coords[1] = B.coords[1] - A.coords[1];
        v1.coords[2] = B.coords[2] - A.coords[2];
        v1.coords[3] = B.coords[3] - A.coords[3];
        v1.coords[4] = B.coords[4] - A.coords[4];

        float modulus = sqrt(v1.coords[0] * v1.coords[0] + v1.coords[1] * v1.coords[1] +
            v1.coords[2] * v1.coords[2] + v1.coords[3] * v1.coords[3] +
            v1.coords[4] * v1.coords[4]);

        if (modulus != 0.0) {
            float factor = 1.0 / modulus;

            // Normalize v1 by dividing by mod(v1)
            v1.coords[0] *= factor;
            v1.coords[1] *= factor;
            v1.coords[2] *= factor;
            v1.coords[3] *= factor;
            v1.coords[4] *= factor;

            // Vector v2 = B - C 
            v2.coords[0] = B.coords[0] - C.coords[0];
            v2.coords[1] = B.coords[1] - C.coords[1];
            v2.coords[2] = B.coords[2] - C.coords[2];
            v2.coords[3] = B.coords[3] - C.coords[3];
            v2.coords[4] = B.coords[4] - C.coords[4];

            modulus = sqrt(v2.coords[0] * v2.coords[0] + v2.coords[1] * v2.coords[1] +
                v2.coords[2] * v2.coords[2] + v2.coords[3] * v2.coords[3] +
                v2.coords[4] * v2.coords[4]);

            if (modulus != 0.0) {
                factor = 1.0 / modulus;

                // Normalize v2 by dividing by mod(v2)
                v2.coords[0] *= factor;
                v2.coords[1] *= factor;
                v2.coords[2] *= factor;
                v2.coords[3] *= factor;
                v2.coords[4] *= factor;

                // Calculate dot product of v1 and v2
                dotProduct = v1.coords[0] * v2.coords[0] + v1.coords[1] * v2.coords[1] +
                    v1.coords[2] * v2.coords[2] + v1.coords[3] * v2.coords[3] +
                    v1.coords[4] * v2.coords[4];
            }
        }

        // Clamp dotProduct to the range [-1, 1]
        dotProduct = fmaxf(fminf(dotProduct, 1.0f), -1.0f);

        // Return the angle in radians
        return acos(dotProduct);
    }
}