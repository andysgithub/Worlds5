#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdio.h>
#include "vector5Double.h"
#include "kernel.cuh"
#include "inline.cuh"

__constant__ RayTracingParams d_params;

__device__ void VectorTrans2(double x, double y, double z, vector5Double* c) {
    for (int i = 0; i < DimTotal; i++) {
        (*c).coords[i] = cudaTrans[i][0] * x +
            cudaTrans[i][1] * y +
            cudaTrans[i][2] * z +
            cudaTrans[i][5];
    }
}

__device__ double vectorAngle(const vector5Double& A, const vector5Double& B, const vector5Double& C) {
    vector5Double v1, v2;
    double dotProduct = 0.0;

    // Vector v1 = B - A 
    v1.coords[0] = B.coords[0] - A.coords[0];
    v1.coords[1] = B.coords[1] - A.coords[1];
    v1.coords[2] = B.coords[2] - A.coords[2];
    v1.coords[3] = B.coords[3] - A.coords[3];
    v1.coords[4] = B.coords[4] - A.coords[4];

    double modulus = sqrt(v1.coords[0] * v1.coords[0] + v1.coords[1] * v1.coords[1] +
        v1.coords[2] * v1.coords[2] + v1.coords[3] * v1.coords[3] +
        v1.coords[4] * v1.coords[4]);

    if (modulus != 0.0) {
        double factor = 1.0 / modulus;

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
    dotProduct = fmax(fmin(dotProduct, 1.0), -1.0);

    // Return the angle in radians
    return acos(dotProduct);
}

// Determine whether nD point c[] in within the set
// Returns true if point is external to the set
__device__ bool ExternalPoint2(vector5Double c, float bailout)
{
    const long MaxCount = (long)(1000);		        // Iteration count for external points
    vector5Double z;										// Temporary 5-D vector
    vector5Double diff;										// Temporary 5-D vector for orbit size
    double ModulusTotal = 0;
    double ModVal = 0;
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

        //    Stop accumulating values when modulus exceeds bailout value
        if (ModVal > bailout)
        {
            count++;
            break;
        }
    }

    // Return true if this point is external to the set
    return (count < MaxCount);
}

__device__ bool ProcessPoint2(float* Modulus, float* Angle, float bailout, vector5Double c) {
    double const PI = 3.1415926536;

    const long MaxCount = 1000;
    vector5Double z;
    vector5Double diff;
    double ModulusTotal = 0;
    double ModVal = 0;
    double AngleTotal = PI;
    long count;

    z.coords[3] = 0;
    z.coords[4] = 0;

    v_mov(c.coords, z.coords);
    vector5Double vectorSet[3];
    v_mov(z.coords, vectorSet[1].coords);

    for (count = 0; count < MaxCount; count++) {
        v_mandel(z.coords, c.coords);
        v_mov(z.coords, vectorSet[2].coords);

        if (count > 0 && count < 10) {
            AngleTotal += vectorAngle(vectorSet[0], vectorSet[1], vectorSet[2]);
        }

        v_subm(c.coords, z.coords, diff.coords);
        ModVal = v_mod(diff.coords);

        ModulusTotal += ModVal;

        if (ModVal > bailout) {
            count++;
            break;
        }

        v_mov(vectorSet[1].coords, vectorSet[0].coords);
        v_mov(vectorSet[2].coords, vectorSet[1].coords);
    }

    *Modulus = (float)(ModulusTotal / count);
    *Angle = (float)(AngleTotal / (count > 10 ? 10 : count + 1));
    return (count < MaxCount);
}

__device__ bool SamplePoint2(double distance, float* Modulus, float* Angle, float bailout, double xFactor, double yFactor, double zFactor, vector5Double c) {
    // Determine the x,y,z coord for this point
    const double XPos = distance * xFactor;
    const double YPos = distance * yFactor;
    const double ZPos = distance * zFactor;

    // Transform 3D point x,y,z into nD fractal space at point c[]
    VectorTrans2(XPos, YPos, ZPos, &c);

    // Determine orbit value for this point
    return ProcessPoint2(Modulus, Angle, bailout, c) ? 1 : 0;
}

__device__ bool SamplePoint2(double distance, float bailout, double xFactor, double yFactor, double zFactor, vector5Double c) {
    // Determine the x,y,z coord for this point
    const double XPos = distance * xFactor;
    const double YPos = distance * yFactor;
    const double ZPos = distance * zFactor;

    // Transform 3D point x,y,z into nD fractal space at point c[]
    VectorTrans2(XPos, YPos, ZPos, &c);

    // Determine orbit value for this point
    return ExternalPoint2(c, bailout) ? 1 : 0;
}

__device__ bool gapFound2(double currentDistance, double surfaceThickness, double xFactor, double yFactor, double zFactor, float bailout, vector5Double c) {
    double testDistance;

    for (int factor = 1; factor <= 4; factor++) {
        testDistance = currentDistance + surfaceThickness * factor / 4;

        if (SamplePoint2(testDistance, bailout, xFactor, yFactor, zFactor, c) == 1) {
            return true;
        }
    }
    return false;
}

__device__ double FindSurface2(double increment, double smoothness, int binarySearchSteps, double currentDistance, double xFactor, double yFactor, double zFactor, float bailout) {
    double stepFactor = smoothness / 10;
    double stepSize = -increment * stepFactor;
    double sampleDistance = currentDistance;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    for (int i = 0; i < binarySearchSteps; i++) {
        sampleDistance += stepSize;

        if (SamplePoint2(sampleDistance, bailout, xFactor, yFactor, zFactor, c) == 0) {
            stepSize = -fabs(stepSize) * stepFactor;
        }
        else {
            stepSize = fabs(stepSize) * stepFactor;
        }
    }
    return sampleDistance;
}

__device__ double FindBoundary2(double increment, int binarySearchSteps, double currentDistance, float previousAngle,
    double boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
    double xFactor, double yFactor, double zFactor, float bailout) {
    double stepSize = -increment / 2;
    double sampleDistance = currentDistance;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    for (int i = 0; i < binarySearchSteps; i++) {
        sampleDistance += stepSize;
        *externalPoint = SamplePoint2(sampleDistance, Modulus, Angle, bailout, xFactor, yFactor, zFactor, c);

        const double angleChange = fabs(*Angle - previousAngle);

        if (angleChange > boundaryInterval) {
            stepSize = -fabs(stepSize) / 2;
        }
        else {
            stepSize = fabs(stepSize) / 2;
        }
    }
    return sampleDistance;
}

__global__ void TraceRayKernel(
    double xFactor, double yFactor, double zFactor,
    int* externalPoints, float* modulusValues, float* angles, double* distances, int* recordedPointsOut) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_params.rayPoints) return;

    float Modulus, Angle;
    double currentDistance = d_params.startDistance;
    double sampleDistance;
    int recordedPoints = 0;
    int sampleCount = 0;
    const vector5Double c = { 0, 0, 0, 0, 0 }; // 5D vector for ray point coordinates

    // Determine orbit value for the starting point
    bool externalPoint = SamplePoint2(currentDistance, &Modulus, &Angle, d_params.bailout, xFactor, yFactor, zFactor, c);

    // Record this point as the first sample
    externalPoints[idx] = externalPoint;
    modulusValues[idx] = Modulus;
    angles[idx] = Angle;
    distances[idx] = currentDistance;
    recordedPoints++;

    // Begin loop
    while (recordedPoints < d_params.rayPoints && sampleCount < d_params.maxSamples) {
        // Move on to the next point
        currentDistance += d_params.increment;
        sampleCount++;

        // Determine orbit properties for this point
        externalPoint = SamplePoint2(currentDistance, &Modulus, &Angle, d_params.bailout, xFactor, yFactor, zFactor, c);

        // If this is an internal point and previous point is external
        if (d_params.activeIndex == 0 && !externalPoint && externalPoints[recordedPoints - 1] == 1) {
            ///// Set value for surface point /////

            // Perform binary search between this and the previous point, to determine surface position
            sampleDistance = FindSurface2(d_params.increment, d_params.smoothness, d_params.binarySearchSteps, currentDistance, xFactor, yFactor, zFactor, d_params.bailout);

            // Test point a short distance further along, to determine whether this is still in the set
            if (d_params.surfaceThickness > 0 && gapFound2(sampleDistance, d_params.surfaceThickness, xFactor, yFactor, zFactor, d_params.bailout, c)) {
                // Back outside the set, so continue as normal for external points
                externalPoint = true;
                continue;
            }
            // Determine orbit properties for this point
            externalPoint = SamplePoint2(sampleDistance, &Modulus, &Angle, d_params.bailout, xFactor, yFactor, zFactor, c);

            // Save this point value in the ray collection
            externalPoints[recordedPoints] = externalPoint ? 1 : 0;
            modulusValues[recordedPoints] = Modulus;
            angles[recordedPoints] = Angle;
            distances[recordedPoints] = sampleDistance;
            recordedPoints++;
        }
        else if (d_params.activeIndex == 1) {
            ///// Set value for external point /////

            double angleChange = fabs(Angle - angles[recordedPoints - 1]);

            // If orbit value is sufficiently different from the last recorded sample
            if (angleChange > d_params.boundaryInterval) {
                // Perform binary search between this and the recorded point, to determine boundary position
                sampleDistance = FindBoundary2(d_params.increment, d_params.binarySearchSteps, currentDistance, angles[recordedPoints - 1],
                    d_params.boundaryInterval, &externalPoint, &Modulus, &Angle,
                    xFactor, yFactor, zFactor, d_params.bailout);

                // Save this point value in the ray collection
                externalPoints[recordedPoints] = externalPoint ? 1 : 0;
                modulusValues[recordedPoints] = Modulus;
                angles[recordedPoints] = Angle;
                distances[recordedPoints] = sampleDistance;
                recordedPoints++;
            }
        }
    }

    distances[recordedPoints] = CUDART_INF;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *recordedPointsOut = recordedPoints + 1;
    }
}

// Host function to initialize the GPU with constant parameters
extern "C" void InitializeGPUKernel(const RayTracingParams * params)
{
    // Copy the parameters to the device's constant memory
    cudaError_t cudaStatus = cudaMemcpyToSymbol(d_params, params, sizeof(RayTracingParams));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error (e.g., throw an exception or return an error code)
    }
}

extern "C" int launchTraceRayKernel(double XFactor, double YFactor, double ZFactor, int rayPoints,
    int* externalPoints, float* modulusValues, float* angles, double* distances)
{
    // Allocate device memory
    int* d_externalPoints, * d_recordedPointsOut;
    float* d_modulusValues, * d_angles;
    double* d_distances;

    cudaMalloc(&d_externalPoints, rayPoints * sizeof(int));
    cudaMalloc(&d_modulusValues, rayPoints * sizeof(float));
    cudaMalloc(&d_angles, rayPoints * sizeof(float));
    cudaMalloc(&d_distances, rayPoints * sizeof(double));
    cudaMalloc(&d_recordedPointsOut, sizeof(int));

    int blockSize = 256;
    int numBlocks = (rayPoints + blockSize - 1) / blockSize;

    // Launch kernel
    TraceRayKernel<<<numBlocks, blockSize>>>(XFactor, YFactor, ZFactor,
        d_externalPoints, d_modulusValues, d_angles, d_distances, d_recordedPointsOut);

    // Copy results back to host
    int recordedPoints;
    cudaMemcpy(&recordedPoints, d_recordedPointsOut, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_externalPoints);
    cudaFree(d_modulusValues);
    cudaFree(d_angles);
    cudaFree(d_distances);
    cudaFree(d_recordedPointsOut);

    return recordedPoints;
}
