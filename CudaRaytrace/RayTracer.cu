#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdio.h>
#include "inline.cuh"
#include "cuda_interface.h"
#include "vectors.cuh"
#include "RayTracer.cuh"

#ifdef __INTELLISENSE__
#define __any_sync(x, y) (y)
#define __activemask() 0
#define CUDART_INF_F 0
#endif

__constant__ float cudaTrans[6][DimTotal];
__constant__ RayTracingParams d_rayParams;
__constant__ RenderingParams d_renderParams;

namespace RayTracer {

    __device__ int TraceRay(float startDistance, Vector3 rayPoint, int rayPoints,
        int* __restrict__ externalPoints, float* __restrict__ modulusValues,
        float* __restrict__ angles, float* __restrict__ distances) {

        const vector5Single c = { 0, 0, 0, 0, 0 };
        float Modulus, Angle, currentDistance = startDistance;
        int recordedPoints = 0, sampleCount = 0;
        bool externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayPoint, c);

        if (recordedPoints < rayPoints) {
            externalPoints[recordedPoints] = externalPoint ? 1 : 0;
            modulusValues[recordedPoints] = Modulus;
            angles[recordedPoints] = Angle;
            distances[recordedPoints] = currentDistance;
            recordedPoints++;
        }

        if (d_rayParams.activeIndex == 0) {
            bool previousPointExternal = true;

            float stepFactor = d_rayParams.surfaceSmoothing / 10;
            float stepSize = -d_rayParams.samplingInterval * stepFactor;

            while (recordedPoints < rayPoints && sampleCount < d_rayParams.maxSamples) {
                currentDistance += d_rayParams.samplingInterval;
                sampleCount++;

                externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayPoint, c);

                bool shouldRecord = !externalPoint && previousPointExternal;

                if (shouldRecord) {
                    float sampleDistance = FindSurface(stepSize, stepFactor, currentDistance, rayPoint);
                    //float sampleDistance = currentDistance;

                    bool foundGap = gapFound(sampleDistance, rayPoint, c);

                    if (d_rayParams.surfaceThickness > 0 && foundGap) {
                        previousPointExternal = true;
                        continue;
                    }
                    externalPoint = SamplePoint(sampleDistance, &Modulus, &Angle, rayPoint, c);

                    externalPoints[recordedPoints] = externalPoint ? 1 : 0;
                    modulusValues[recordedPoints] = Modulus;
                    angles[recordedPoints] = Angle;
                    distances[recordedPoints] = sampleDistance;
                    recordedPoints++;
                }

                previousPointExternal = externalPoint;
            }
        }

        if (d_rayParams.activeIndex == 1) {
            while (recordedPoints < rayPoints && sampleCount < d_rayParams.maxSamples) {
                currentDistance += d_rayParams.samplingInterval;
                sampleCount++;

                externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayPoint, c);

                ///// Set value for external point /////

                float angleChange = fabs(Angle - angles[recordedPoints - 1]);

                // If orbit value is sufficiently different from the last recorded sample
                if (angleChange > d_rayParams.boundaryInterval) {
                    // Perform binary search between this and the recorded point, to determine boundary position
                    float sampleDistance = FindBoundary(currentDistance, angles[recordedPoints - 1],
                        &externalPoint, &Modulus, &Angle, rayPoint);

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
        float stepSize, float stepFactor, float currentDistance,
        Vector3 rayPoint) {

        float sampleDistance = currentDistance;
        const vector5Single c = { 0, 0, 0, 0, 0 };

        for (int i = 0; i < d_rayParams.binarySearchSteps; i++) {
            sampleDistance += stepSize;
            stepSize = fabs(stepSize) * stepFactor;

            bool isExternal = SamplePoint(sampleDistance, rayPoint, c);

            // If inside the fractal, step back next time
            if (!isExternal) {
                stepSize = -stepSize;
            }
        }
        return sampleDistance;
    }

    __device__ float FindBoundary(float currentDistance, float previousAngle,
        bool* externalPoint, float* Modulus, float* Angle,
        Vector3 rayPoint) {

        float stepSize = -d_rayParams.samplingInterval / 2;
        float sampleDistance = currentDistance;
        const vector5Single c = { 0, 0, 0, 0, 0 };

        #pragma unroll 1
        for (int i = 0; i < d_rayParams.binarySearchSteps; i++) {
            sampleDistance += stepSize;
            *externalPoint = SamplePoint(sampleDistance, Modulus, Angle, rayPoint, c);

            float angleChange = fabsf(*Angle - previousAngle);
            bool exceedsBoundary = (angleChange > d_rayParams.boundaryInterval);

            // Use a branchless approach to update stepSize
            float stepSizeAbs = fabsf(stepSize);
            stepSize = copysignf(stepSizeAbs / 2, exceedsBoundary ? -stepSizeAbs : stepSizeAbs);
        }

        return sampleDistance;
    }

    __device__ bool SamplePoint(float distance, float* Modulus, float* Angle, Vector3 rayPoint, vector5Single c) {
        // Determine the x,y,z coord for this point
        Vector3 imagePoint = Vector3(distance * rayPoint.X, distance * rayPoint.Y, distance * rayPoint.Z);

        // Transform 3D point x,y,z into nD fractal space at point c[]
        VectorTrans(imagePoint, &c);

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

        float bailout_squared = d_rayParams.bailout * d_rayParams.bailout;
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

    __device__ bool SamplePoint(float distance, Vector3 rayPoint, vector5Single c) {
        // Determine the x,y,z coord for this point
        Vector3 imagePoint = Vector3(distance * rayPoint.X, distance * rayPoint.Y, distance * rayPoint.Z);

        // Transform 3D point x,y,z into nD fractal space at point c[]
        VectorTrans(imagePoint, &c);

        // Determine orbit value for this point
        constexpr int MaxCount = 1000;  // Use int instead of long for better performance on GPUs
        vector5Single z = { 0 };
        vector5Single diff;
        float ModulusTotal = 0;
        float bailout_squared = d_rayParams.bailout * d_rayParams.bailout;

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

    __device__ bool gapFound(float currentDistance, Vector3 rayPoint, vector5Single c) {
        float testDistance;

        #pragma unroll
        for (int factor = 1; factor <= 4; factor++) {
            testDistance = currentDistance + d_rayParams.surfaceThickness * factor / 4;
            if (SamplePoint(testDistance, rayPoint, c)) {
                return true;
            }
        }

        return false;
    }

    __device__ void VectorTrans(Vector3 imagePoint, vector5Single* c) {
        for (int col = 0; col < DimTotal; col++) {
            (*c).coords[col] =
                cudaTrans[0][col] * imagePoint.X +
                cudaTrans[1][col] * imagePoint.Y +
                cudaTrans[2][col] * imagePoint.Z +
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