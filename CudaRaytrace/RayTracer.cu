#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdio.h>
#include "inline.cuh"
#include "cuda_interface.h"
#include "RayTracer.cuh"

#ifdef __INTELLISENSE__
#define __any_sync(x, y) (y)
#define __activemask() 0
#define CUDART_INF_F 0
#endif

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

            //float stepFactor = rayParams->surfaceSmoothing / 10;
            //float stepSize = -rayParams->samplingInterval * stepFactor;

            while (recordedPoints < rayPoints && sampleCount < rayParams->maxSamples) {
                currentDistance += rayParams->samplingInterval;
                sampleCount++;

                externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayParams->bailout, xFactor, yFactor, zFactor, c);

                bool shouldRecord = !externalPoint && previousPointExternal;

                if (shouldRecord) {
                    //float sampleDistance = FindSurface(
                    //    stepSize, stepFactor, rayParams->binarySearchSteps,
                    //    currentDistance, xFactor, yFactor, zFactor, rayParams->bailout);
                    float sampleDistance = currentDistance;

                    bool foundGap = gapFound(sampleDistance, rayParams->surfaceThickness, xFactor, yFactor, zFactor, rayParams->bailout, c);

                    if (rayParams->surfaceThickness > 0 && foundGap) {
                        previousPointExternal = true;
                        continue;
                    }
                    externalPoint = SamplePoint(sampleDistance, &Modulus, &Angle, rayParams->bailout, xFactor, yFactor, zFactor, c);

                    externalPoints[recordedPoints] = externalPoint ? 1 : 0;
                    modulusValues[recordedPoints] = Modulus;
                    angles[recordedPoints] = Angle;
                    distances[recordedPoints] = sampleDistance;
                    recordedPoints++;
                }

                previousPointExternal = externalPoint;
            }
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
        
        if (recordedPoints < rayPoints) {
            distances[recordedPoints] = CUDART_INF_F;
        }
        return recordedPoints + 1;
    }

    // Perform a binary search to refine the surface position
    __device__ float FindSurface(
        float stepSize, float stepFactor, int binarySearchSteps, float currentDistance,
        float xFactor, float yFactor, float zFactor, float bailout) {

        float sampleDistance = currentDistance;
        const vector5Single c = { 0, 0, 0, 0, 0 };

        for (int i = 0; i < binarySearchSteps; i++) {
            sampleDistance += stepSize;
            stepSize = fabs(stepSize) * stepFactor;

            bool isExternal = SamplePoint(sampleDistance, bailout, xFactor, yFactor, zFactor, c);

            // If inside the fractal, step back next time
            if (!isExternal) {
                stepSize = -stepSize;
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

        #pragma unroll 1
        for (int i = 0; i < binarySearchSteps; i++) {
            sampleDistance += stepSize;
            *externalPoint = SamplePoint(sampleDistance, Modulus, Angle, bailout, xFactor, yFactor, zFactor, c);

            float angleChange = fabsf(*Angle - previousAngle);
            bool exceedsBoundary = (angleChange > boundaryInterval);

            // Use a branchless approach to update stepSize
            float stepSizeAbs = fabsf(stepSize);
            stepSize = copysignf(stepSizeAbs / 2, exceedsBoundary ? -stepSizeAbs : stepSizeAbs);
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

        constexpr float PI = 3.1415926536f;
        constexpr int MaxCount = 100;
        vector5Single z = { 0 };
        vector5Single diff;
        float ModulusTotal = 0;
        float AngleTotal = PI;

        z.coords[DimTotal - 2] = 0;
        z.coords[DimTotal - 1] = 0;
        v_mov(c.coords, z.coords);

        vector5Single vectorSet[3];
        v_mov(z.coords, vectorSet[1].coords);

        float bailout_squared = bailout * bailout;
        int count = 0;
        bool escaped = false;

        #pragma unroll 1
        for (; count < MaxCount; count++) {
            v_mandel(z.coords, c.coords);
            v_mov(z.coords, vectorSet[2].coords);

            if (count > 0 && count < 10) {
                AngleTotal += vectorAngle(vectorSet[0], vectorSet[1], vectorSet[2]);
            }

            v_subm(c.coords, z.coords, diff.coords);
            float ModVal = v_mod(diff.coords);
            ModulusTotal += ModVal;

            escaped = (ModVal > bailout_squared);
            if (__any_sync(__activemask(), escaped)) {
                if (!escaped) count = MaxCount;
                break;
            }

            v_mov(vectorSet[1].coords, vectorSet[0].coords);
            v_mov(vectorSet[2].coords, vectorSet[1].coords);
        }

        *Modulus = ModulusTotal / (count + 1);
        *Angle = AngleTotal / (count < 10 ? count + 1 : 10);

        return escaped;
    }

    __device__ bool SamplePoint(float distance, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c) {
        // Determine the x,y,z coord for this point
        const float XPos = distance * xFactor;
        const float YPos = distance * yFactor;
        const float ZPos = distance * zFactor;

        // Transform 3D point x,y,z into nD fractal space at point c[]
        VectorTrans(XPos, YPos, ZPos, &c);

        // Determine orbit value for this point
        constexpr int MaxCount = 1000;  // Use int instead of long for better performance on GPUs
        vector5Single z = { 0 };
        vector5Single diff;
        float ModulusTotal = 0;
        float bailout_squared = bailout * bailout;

        z.coords[DimTotal - 2] = 0;
        z.coords[DimTotal - 1] = 0;
        v_mov(c.coords, z.coords);

        #pragma unroll 1
        for (int count = 0; count < MaxCount; ++count) {
            v_mandel(z.coords, c.coords);  // z = z*z + c

            // Determine modulus for this point in orbit
            v_subm(c.coords, z.coords, diff.coords);  // Current orbit size = mod(z - c)
            float ModVal = v_mod(diff.coords);

            // Check if point has escaped
            if (ModVal > bailout_squared) return true;

            // Accumulate modulus value
            ModulusTotal += ModVal;
        }

        return false;
    }

    __device__ bool gapFound(float currentDistance, float surfaceThickness, float xFactor, float yFactor, float zFactor, float bailout, vector5Single c) {
        float testDistance;

        #pragma unroll
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
        float v1[5], v2[5];
        float dot1 = 0.0f, dot2 = 0.0f, dotProduct = 0.0f;

        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            v1[i] = B.coords[i] - A.coords[i];
            v2[i] = B.coords[i] - C.coords[i];
            dot1 += v1[i] * v1[i];
            dot2 += v2[i] * v2[i];
        }

        // Use a small epsilon value to avoid division by zero
        const float epsilon = 1e-6f;
        float invMod1 = 1.0f / sqrtf(fmaxf(dot1, epsilon));
        float invMod2 = 1.0f / sqrtf(fmaxf(dot2, epsilon));

        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            dotProduct += (v1[i] * invMod1) * (v2[i] * invMod2);
        }

        // Clamp dotProduct to [-1, 1] range
        dotProduct = fmaxf(-1.0f, fminf(1.0f, dotProduct));
        return acosf(dotProduct);
    }
}