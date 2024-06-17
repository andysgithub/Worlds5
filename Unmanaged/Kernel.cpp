#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Define the vector5Double structure
struct vector5Double {
    double x1, x2, x3, x4, x5;
};

__device__ void VectorTrans(double XPos, double YPos, double ZPos, vector5Double* c);
__device__ bool ProcessPoint(float* Modulus, float* Angle, vector5Double c);
__device__ bool SamplePoint(double distance, float* Modulus, float* Angle, double xFactor, double yFactor, double zFactor, vector5Double c);

__device__ bool gapFound(double currentDistance, double surfaceThickness, double xFactor, double yFactor, double zFactor, vector5Double c)
{
    double testDistance;

    for (int factor = 1; factor <= 4; factor++)
    {
        testDistance = currentDistance + surfaceThickness * factor / 4;

        if (SamplePoint(testDistance, nullptr, nullptr, xFactor, yFactor, zFactor, c) == 1)
        {
            return true;
        }
    }
    return false;
}

__device__ double FindSurface(double increment, double smoothness, int binarySearchSteps, double currentDistance, double xFactor, double yFactor, double zFactor)
{
    double stepFactor = smoothness / 10;
    double stepSize = -increment * stepFactor;
    double sampleDistance = currentDistance;
    const vector5Double c = { 0, 0, 0, 0, 0 };

    // Perform binary search between the current and previous points, to determine boundary position
    for (int i = 0; i < binarySearchSteps; i++)
    {
        // Step back or forwards by half the distance
        sampleDistance += stepSize;

        // If this point is internal to the set
        if (SamplePoint(sampleDistance, nullptr, nullptr, xFactor, yFactor, zFactor, c) == 0)
        {
            // Step back next time
            stepSize = -fabs(stepSize) * stepFactor;
        }
        else
        {
            // Step forward next time
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
    const vector5Double c = { 0, 0, 0, 0, 0 };  // 5D vector for ray point coordinates

    // Perform binary search between the current and previous points, to determine boundary position
    for (int i = 0; i < binarySearchSteps; i++) {
        // Step back or forwards by half the distance
        sampleDistance += stepSize;
        // Take a sample at this point
        *externalPoint = SamplePoint(sampleDistance, Modulus, Angle, xFactor, yFactor, zFactor, c);

        const double angleChange = fabs(*Angle - previousAngle);

        // If this point is sufficiently different from the last recorded sample
        if (angleChange > boundaryInterval) {
            // Step back next time
            stepSize = -fabs(stepSize) / 2;
        }
        else {
            // Step forward next time
            stepSize = fabs(stepSize) / 2;
        }
    }
    return sampleDistance;
}

__global__ void TraceRayKernel(double startDistance, double increment, double smoothness, double surfaceThickness,
    double xFactor, double yFactor, double zFactor,
    int* externalPoints, float* modulusValues, float* angles, double* distances,
    int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
    int activeIndex)
{
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

    while (recordedPoints < rayPoints && sampleCount < maxSamples)
    {
        currentDistance += increment;
        sampleCount++;

        externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, c);

        if (activeIndex == 0 && externalPoint == 0 && externalPoints[recordedPoints - 1] == 1)
        {
            sampleDistance = FindSurface(increment, smoothness, binarySearchSteps, currentDistance, xFactor, yFactor, zFactor);

            if (surfaceThickness > 0 && gapFound(sampleDistance, surfaceThickness, xFactor, yFactor, zFactor, c))
            {
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
        else if (activeIndex == 1)
        {
            double angleChange = fabs(Angle - angles[recordedPoints - 1]);

            if (angleChange > boundaryInterval)
            {
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