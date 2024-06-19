#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <cmath>
#include "kernel.cuh"

__device__ void VectorTrans(double x, double y, double z, vector5Double* c) {
    for (int i = 0; i < DimTotal; i++) {
        (*c).coords[i] = m_Trans[i][0] * x +
            m_Trans[i][1] * y +
            m_Trans[i][2] * z +
            m_Trans[i][5];
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

__device__ bool ProcessPoint(float* Modulus, float* Angle, vector5Double c, float bailout) {
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

__device__ bool SamplePoint(double distance, float* Modulus, float* Angle, double xFactor, double yFactor, double zFactor, float bailout, vector5Double c) {
    const double XPos = distance * xFactor;
    const double YPos = distance * yFactor;
    const double ZPos = distance * zFactor;

    VectorTrans(XPos, YPos, ZPos, &c);
    return ProcessPoint(Modulus, Angle, c, bailout) ? 1 : 0;
}

__device__ bool gapFound(double currentDistance, double surfaceThickness, double xFactor, double yFactor, double zFactor, float bailout, vector5Double c) {
    double testDistance;

    for (int factor = 1; factor <= 4; factor++) {
        testDistance = currentDistance + surfaceThickness * factor / 4;

        if (SamplePoint(testDistance, nullptr, nullptr, xFactor, yFactor, zFactor, bailout, c) == 1) {
            return true;
        }
    }
    return false;
}

__device__ double FindSurface(double increment, double smoothness, int binarySearchSteps, double currentDistance, double xFactor, double yFactor, double zFactor, float bailout) {
    double stepFactor = smoothness / 10;
    double stepSize = -increment * stepFactor;
    double sampleDistance = currentDistance;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    for (int i = 0; i < binarySearchSteps; i++) {
        sampleDistance += stepSize;

        if (SamplePoint(sampleDistance, nullptr, nullptr, xFactor, yFactor, zFactor, bailout, c) == 0) {
            stepSize = -fabs(stepSize) * stepFactor;
        }
        else {
            stepSize = fabs(stepSize) * stepFactor;
        }
    }
    return sampleDistance;
}

__device__ double FindBoundary(double increment, int binarySearchSteps, double currentDistance, float previousAngle,
    double boundaryInterval, bool* externalPoint, float* Modulus, float* Angle,
    double xFactor, double yFactor, double zFactor, float bailout) {
    double stepSize = -increment / 2;
    double sampleDistance = currentDistance;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    for (int i = 0; i < binarySearchSteps; i++) {
        sampleDistance += stepSize;
        *externalPoint = SamplePoint(sampleDistance, Modulus, Angle, xFactor, yFactor, zFactor, bailout, c);

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
    cudaMemcpyToSymbol(m_Trans, h_Trans, sizeof(double) * DimTotal * 6);
}

__global__ void TraceRayKernel(
    double startDistance, double increment, double smoothness, double surfaceThickness,
    double xFactor, double yFactor, double zFactor, float bailout,
    int* externalPoints, float* modulusValues, float* angles, double* distances,
    int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
    int activeIndex
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= rayPoints) return;

    float Modulus, Angle;
    double currentDistance = startDistance;
    double sampleDistance;
    int recordedPoints = 0;
    int sampleCount = 0;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    bool externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, bailout, c);

    externalPoints[idx] = externalPoint;
    modulusValues[idx] = Modulus;
    angles[idx] = Angle;
    distances[idx] = currentDistance;
    recordedPoints++;

    while (recordedPoints < rayPoints && sampleCount < maxSamples) {
        currentDistance += increment;
        sampleCount++;

        externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, bailout, c);

        if (activeIndex == 0 && externalPoint == 0 && externalPoints[recordedPoints - 1] == 1) {
            sampleDistance = FindSurface(increment, smoothness, binarySearchSteps, currentDistance, xFactor, yFactor, zFactor);

            if (surfaceThickness > 0 && gapFound(sampleDistance, surfaceThickness, xFactor, yFactor, zFactor, bailout, c)) {
                externalPoint = true;
                continue;
            }
            externalPoint = SamplePoint(sampleDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, bailout, c);

            externalPoints[recordedPoints] = externalPoint;
            modulusValues[recordedPoints] = Modulus;
            angles[recordedPoints] = Angle;
            distances[recordedPoints] = sampleDistance;
            recordedPoints++;
        }
        else if (activeIndex == 1) {
            double angleChange = fabs(Angle - angles[recordedPoints - 1]);

            if (angleChange > boundaryInterval) {
                sampleDistance = FindBoundary(increment, binarySearchSteps, currentDistance, angles[recordedPoints - 1],
                    boundaryInterval, &externalPoint, &Modulus, &Angle,
                    xFactor, yFactor, zFactor);

                externalPoints[recordedPoints] = externalPoint;
                modulusValues[recordedPoints] = Modulus;
                angles[recordedPoints] = Angle;
                distances[recordedPoints] = sampleDistance;
                recordedPoints++;
            }
        }
    }

    distances[recordedPoints] = HUGE_VAL;
}

// Example function to launch the kernel
void runKernel() {
    // Define the number of rays and other parameters
    int rayPoints = 1024;
    int maxSamples = 10000;
    double startDistance = 0.0;
    double increment = 0.1;
    double smoothness = 0.5;
    double surfaceThickness = 0.1;
    double xFactor = 1.0;
    double yFactor = 1.0;
    double zFactor = 1.0;
    double boundaryInterval = 0.01;
    int binarySearchSteps = 10;
    int activeIndex = 0;

    // Allocate memory for device arrays
    int* d_externalPoints;
    float* d_modulusValues;
    float* d_angles;
    double* d_distances;

    cudaMalloc(&d_externalPoints, rayPoints * sizeof(int));
    cudaMalloc(&d_modulusValues, rayPoints * sizeof(float));
    cudaMalloc(&d_angles, rayPoints * sizeof(float));
    cudaMalloc(&d_distances, rayPoints * sizeof(double));

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (rayPoints + blockSize - 1) / blockSize;

    // Launch the kernel
    TraceRayKernel<<<numBlocks, blockSize>>>(startDistance, increment, smoothness, surfaceThickness,
        xFactor, yFactor, zFactor,
        d_externalPoints, d_modulusValues, d_angles, d_distances,
        rayPoints, maxSamples, boundaryInterval, binarySearchSteps,
        activeIndex);

    // Synchronize to check for errors
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_externalPoints);
    cudaFree(d_modulusValues);
    cudaFree(d_angles);
    cudaFree(d_distances);
}
