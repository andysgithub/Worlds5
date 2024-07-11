#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdio.h>
#include "inline.cuh"
#include "cuda_interface.h"
#include "RayTracer.cuh"

__constant__ RayTracingParams d_params;

__global__ void TraceRayKernel(
    float xFactor, float yFactor, float zFactor, int rayPoints,
    int* externalPoints, float* modulusValues, float* angles, float* distances, int* recordedPointsOut) {

    float Modulus, Angle;
    float currentDistance = d_params.sphereRadius;
    float sampleDistance;
    int recordedPoints = 0;
    int sampleCount = 0;
    const vector5Single c = { 0, 0, 0, 0, 0 }; // 5D vector for ray point coordinates

    // Determine orbit value for the starting point
    bool externalPoint = RayTracer::SamplePoint2(currentDistance, &Modulus, &Angle, d_params.bailout, xFactor, yFactor, zFactor, c);

    // Record this point as the first sample
    externalPoints[recordedPoints] = externalPoint ? 1 : 0;
    modulusValues[recordedPoints] = Modulus;
    angles[recordedPoints] = Angle;
    distances[recordedPoints] = currentDistance;
    recordedPoints++;

    // Begin loop
    while (recordedPoints < rayPoints && sampleCount < d_params.maxSamples) {
        // Move on to the next point
        currentDistance += d_params.samplingInterval;
        sampleCount++;

        // Determine orbit properties for this point
        externalPoint = RayTracer::SamplePoint2(currentDistance, &Modulus, &Angle, d_params.bailout, xFactor, yFactor, zFactor, c);

        // If this is an internal point and previous point is external
        if (d_params.activeIndex == 0 && !externalPoint && externalPoints[recordedPoints - 1] == 1) {
            ///// Set value for surface point /////

            // Perform binary search between this and the previous point, to determine surface position
            sampleDistance = RayTracer::FindSurface2(d_params.samplingInterval, d_params.surfaceSmoothing, d_params.binarySearchSteps, currentDistance, xFactor, yFactor, zFactor, d_params.bailout);
            bool foundGap = RayTracer::gapFound2(sampleDistance, d_params.surfaceThickness, xFactor, yFactor, zFactor, d_params.bailout, c);

            // Test point a short distance further along, to determine whether this is still in the set
            if (d_params.surfaceThickness > 0 && RayTracer::gapFound2(sampleDistance, d_params.surfaceThickness, xFactor, yFactor, zFactor, d_params.bailout, c)) {
                // Back outside the set, so continue as normal for external points
                externalPoint = true;
                continue;
            }
            // Determine orbit properties for this point
            externalPoint = RayTracer::SamplePoint2(sampleDistance, &Modulus, &Angle, d_params.bailout, xFactor, yFactor, zFactor, c);

            // Save this point value in the ray collection
            externalPoints[recordedPoints] = externalPoint ? 1 : 0;
            modulusValues[recordedPoints] = Modulus;
            angles[recordedPoints] = Angle;
            distances[recordedPoints] = sampleDistance;
            recordedPoints++;
        }
        else if (d_params.activeIndex == 1) {
            ///// Set value for external point /////

            float angleChange = fabs(Angle - angles[recordedPoints - 1]);

            // If orbit value is sufficiently different from the last recorded sample
            if (angleChange > d_params.boundaryInterval) {
                // Perform binary search between this and the recorded point, to determine boundary position
                sampleDistance = RayTracer::FindBoundary2(d_params.samplingInterval, d_params.binarySearchSteps, currentDistance, angles[recordedPoints - 1],
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
    *recordedPointsOut = recordedPoints + 1;
}

// Host function to initialize the GPU with constant parameters
extern "C" cudaError_t InitializeGPUKernel(const RayTracingParams* params)
{
    // Copy the parameters to the device's constant memory
    return cudaMemcpyToSymbol((const void*)&d_params, (const void*)params, sizeof(RayTracingParams));

    void* d_addr;
    cudaError_t error;

    // Get the address of the symbol in device memory
    error = cudaGetSymbolAddress(&d_addr, (const void*)&d_params);
    if (error != cudaSuccess) {
        return error;
    }

    // Copy the data to the symbol
    error = cudaMemcpy(d_addr, params, sizeof(RayTracingParams), cudaMemcpyHostToDevice);
    return error;
}

extern "C" cudaError_t InitializeTransformMatrix(const float* positionMatrix)
{
    return cudaMemcpyToSymbol(cudaTrans, positionMatrix, sizeof(float) * DimTotal * (DimTotal + 1));
}

extern "C" int launchTraceRayKernel(float XFactor, float YFactor, float ZFactor, int rayPoints,
    int* externalPoints, float* modulusValues, float* angles, float* distances)
{
    // Allocate device memory
    int* d_externalPoints, * d_recordedPointsOut;
    float* d_modulusValues, * d_angles;
    float* d_distances;

    cudaMalloc(&d_externalPoints, rayPoints * sizeof(int));
    cudaMalloc(&d_modulusValues, rayPoints * sizeof(float));
    cudaMalloc(&d_angles, rayPoints * sizeof(float));
    cudaMalloc(&d_distances, rayPoints * sizeof(float));
    cudaMalloc(&d_recordedPointsOut, sizeof(int));

    // Launch kernel
    TraceRayKernel<<<1, 1>>>(XFactor, YFactor, ZFactor, rayPoints,
        d_externalPoints, d_modulusValues, d_angles, d_distances, d_recordedPointsOut);

    // Copy results back to host
    int recordedPoints;
    cudaMemcpy(externalPoints, d_externalPoints, rayPoints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(modulusValues, d_modulusValues, rayPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(angles, d_angles, rayPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, d_distances, rayPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&recordedPoints, d_recordedPointsOut, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_externalPoints);
    cudaFree(d_modulusValues);
    cudaFree(d_angles);
    cudaFree(d_distances);
    cudaFree(d_recordedPointsOut);

    return recordedPoints;
}