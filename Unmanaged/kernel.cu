#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

__device__ bool ProcessPoint(float* Modulus, float* Angle, vector5Double c) {
    double const PI = 3.1415926536;
    double const PI_OVER_2 = PI / 2;

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

        if (ModVal > m_Bailout) {
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

__device__ bool SamplePoint(double distance, float* Modulus, float* Angle, double xFactor, double yFactor, double zFactor, vector5Double c) {
    const double XPos = distance * xFactor;
    const double YPos = distance * yFactor;
    const double ZPos = distance * zFactor;

    VectorTrans(XPos, YPos, ZPos, &c);
    return ProcessPoint(Modulus, Angle, c) ? 1 : 0;
}

__device__ double FindSurface(double increment, double smoothness, int binarySearchSteps, double currentDistance, double xFactor, double yFactor, double zFactor) {
    double stepFactor = smoothness / 10;
    double stepSize = -increment * stepFactor;
    double sampleDistance = currentDistance;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    for (int i = 0; i < binarySearchSteps; i++) {
        sampleDistance += stepSize;

        if (SamplePoint(sampleDistance, nullptr, nullptr, xFactor, yFactor, zFactor, c) == 0) {
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
    double xFactor, double yFactor, double zFactor) {
    double stepSize = -increment / 2;
    double sampleDistance = currentDistance;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    for (int i = 0; i < binarySearchSteps; i++) {
        sampleDistance += stepSize;
        *externalPoint = SamplePoint(sampleDistance, Modulus, Angle, xFactor, yFactor, zFactor, c);

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

__global__ void TraceRayKernel(double startDistance, double increment, double smoothness, double surfaceThickness,
    double xFactor, double yFactor, double zFactor,
    int* externalPoints, float* modulusValues, float* angles, double* distances,
    int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
    int activeIndex) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= rayPoints) return;

    float Modulus, Angle;
    double currentDistance = startDistance;
    double sampleDistance;
    int recordedPoints = 0;
    int sampleCount = 0;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    bool externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, c);

    externalPoints[idx] = externalPoint;
    modulusValues[idx] = Modulus;
    angles[idx] = Angle;
    distances[idx] = currentDistance;
    recordedPoints++;

    while (recordedPoints < rayPoints && sampleCount < maxSamples) {
        currentDistance += increment;
        sampleCount++;

        externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, c);

        if (activeIndex == 0 && externalPoint == 0 && externalPoints[recordedPoints - 1] == 1) {
            sampleDistance = FindSurface(increment, smoothness, binarySearchSteps, currentDistance, xFactor, yFactor, zFactor);

            if (surfaceThickness > 0 && gapFound(sampleDistance, surfaceThickness, xFactor, yFactor, zFactor, c)) {
                externalPoint = true;
                continue;
            }
            externalPoint = SamplePoint(sampleDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, c);

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

void copyTransformationMatrixToDevice(double h_Trans[DimTotal][6]) {
    cudaMemcpyToSymbol(m_Trans, h_Trans, sizeof(double) * DimTotal * 6);
}
