#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdio.h>
#include "vector5Double.h"
#include "kernel.cuh"
#include "inline.cuh"

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

__device__ bool ProcessPoint(float* Modulus, float* Angle, float bailout, vector5Double c) {
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
    const double XPos = distance * xFactor;
    const double YPos = distance * yFactor;
    const double ZPos = distance * zFactor;

    VectorTrans2(XPos, YPos, ZPos, &c);
    return ProcessPoint(Modulus, Angle, bailout, c) ? 1 : 0;
}

__device__ bool gapFound2(double currentDistance, double surfaceThickness, double xFactor, double yFactor, double zFactor, float bailout, vector5Double c) {
    double testDistance;

    for (int factor = 1; factor <= 4; factor++) {
        testDistance = currentDistance + surfaceThickness * factor / 4;

        if (SamplePoint2(testDistance, nullptr, nullptr, bailout, xFactor, yFactor, zFactor, c) == 1) {
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

        if (SamplePoint2(sampleDistance, nullptr, nullptr, bailout, xFactor, yFactor, zFactor, c) == 0) {
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

void copyTransformationMatrixToDevice(double h_Trans[DimTotal][6]) {
    cudaMemcpyToSymbol(cudaTrans, h_Trans, sizeof(double) * DimTotal * 6);
}

__global__ void TraceRayKernel(
    double startDistance, double increment, double smoothness, double surfaceThickness,
    double xFactor, double yFactor, double zFactor, float bailout,
    int* externalPoints, float* modulusValues, float* angles, double* distances,
    int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
    int activeIndex, int* recordedPointsOut) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= rayPoints) return;

    printf("Index: %d\n", idx);

    float Modulus, Angle;
    double currentDistance = startDistance;
    double sampleDistance;
    int recordedPoints = 0;
    int sampleCount = 0;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    bool externalPoint = SamplePoint2(currentDistance, &Modulus, &Angle, bailout, xFactor, yFactor, zFactor, c);

    externalPoints[idx] = externalPoint;
    modulusValues[idx] = Modulus;
    angles[idx] = Angle;
    distances[idx] = currentDistance;
    recordedPoints++;

    while (recordedPoints < rayPoints && sampleCount < maxSamples) {
        currentDistance += increment;
        sampleCount++;

        externalPoint = SamplePoint2(currentDistance, &Modulus, &Angle, bailout, xFactor, yFactor, zFactor, c);

        if (activeIndex == 0 && externalPoint == 0 && externalPoints[recordedPoints - 1] == 1) {
            sampleDistance = FindSurface2(increment, smoothness, binarySearchSteps, currentDistance, xFactor, yFactor, zFactor, bailout);

            if (surfaceThickness > 0 && gapFound2(sampleDistance, surfaceThickness, xFactor, yFactor, zFactor, bailout, c)) {
                externalPoint = true;
                continue;
            }
            externalPoint = SamplePoint2(sampleDistance, &Modulus, &Angle, bailout, xFactor, yFactor, zFactor, c);

            externalPoints[recordedPoints] = externalPoint;
            modulusValues[recordedPoints] = Modulus;
            angles[recordedPoints] = Angle;
            distances[recordedPoints] = sampleDistance;
            recordedPoints++;
        }
        else if (activeIndex == 1) {
            double angleChange = fabs(Angle - angles[recordedPoints - 1]);

            if (angleChange > boundaryInterval) {
                sampleDistance = FindBoundary2(increment, binarySearchSteps, currentDistance, angles[recordedPoints - 1],
                    boundaryInterval, &externalPoint, &Modulus, &Angle,
                    xFactor, yFactor, zFactor, bailout);

                externalPoints[recordedPoints] = externalPoint;
                modulusValues[recordedPoints] = Modulus;
                angles[recordedPoints] = Angle;
                distances[recordedPoints] = sampleDistance;
                recordedPoints++;
            }
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *recordedPointsOut = recordedPoints + 1;
    }

    distances[recordedPoints] = CUDART_INF;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *recordedPointsOut = recordedPoints + 1;
    }
}

extern "C" int launchTraceRayKernel(double startDistance, double increment, double smoothness, double surfaceThickness,
    double XFactor, double YFactor, double ZFactor, float bailout,
    int* externalPoints, float* modulusValues, float* angles, double* distances,
    int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
    int activeIndex)
{
    // Allocate device memory
    int* d_externalPoints, * d_recordedPointsOut;
    float* d_modulusValues, * d_angles;
    double* d_distances;

    cudaMalloc(&d_externalPoints, rayPoints * sizeof(int));
    cudaMalloc(&d_modulusValues, rayPoints * sizeof(float));
    cudaMalloc(&d_angles, rayPoints * sizeof(float));
    cudaMalloc(&d_distances, (rayPoints + 1) * sizeof(double));
    cudaMalloc(&d_recordedPointsOut, sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_externalPoints, externalPoints, rayPoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_modulusValues, modulusValues, rayPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_angles, angles, rayPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, distances, rayPoints * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (rayPoints + blockSize - 1) / blockSize;

    // Launch kernel
    TraceRayKernel<<<numBlocks, blockSize >>>(startDistance, increment, smoothness, surfaceThickness,
        XFactor, YFactor, ZFactor, bailout,
        d_externalPoints, d_modulusValues, d_angles, d_distances,
        rayPoints, maxSamples, boundaryInterval, binarySearchSteps,
        activeIndex, d_recordedPointsOut);

    // Copy results back to host
    cudaMemcpy(externalPoints, d_externalPoints, rayPoints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulusValues, d_modulusValues, rayPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(angles, d_angles, rayPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, d_distances, (rayPoints + 1) * sizeof(double), cudaMemcpyDeviceToHost);

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

__global__ void ProcessPointKernel(float* d_Modulus, float* d_Angle, float bailout, vector5Double* d_c, bool* d_result) {
    *d_result = ProcessPoint(d_Modulus, d_Angle, bailout, *d_c);
}

extern "C" void launchProcessPointKernel(float* d_Modulus, float* d_Angle, float bailout, vector5Double* d_c, bool* d_result)
{
    ProcessPointKernel<<<1, 1>>>(d_Modulus, d_Angle, bailout, d_c, d_result);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
}

__global__ void SamplePointKernel(double distance, float* d_Modulus, float* d_Angle, float bailout,
    double xFactor, double yFactor, double zFactor, vector5Double c, bool* d_result)
{
    *d_result = SamplePoint2(distance, d_Modulus, d_Angle, bailout, xFactor, yFactor, zFactor, c);
}

extern "C" bool launchSamplePointKernel(double distance, float* d_Modulus, float* d_Angle, float bailout,
    double xFactor, double yFactor, double zFactor, vector5Double * d_c)
{
    bool* d_result;
    bool h_result;

    cudaMalloc((void**)&d_result, sizeof(bool));

    vector5Double h_c;
    cudaMemcpy(&h_c, d_c, sizeof(vector5Double), cudaMemcpyDeviceToHost);

    SamplePointKernel<<<1, 1>>>(distance, d_Modulus, d_Angle, bailout, xFactor, yFactor, zFactor, h_c, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return h_result;
}

