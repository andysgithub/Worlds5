#include "stdafx.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include "RayProcessing.h"

#include <cuda_runtime.h>
#include "Parameters.h"
#include "Vectors.h"

const BYTE MAX_COLOURS = 255;

// Produce the collection of fractal point values for the given vector
int TraceRay(float startDistance, RayTracingParams rayParams, Vector3 rayPoint,
    int externalPoints[], float modulusValues[], float angles[], float distances[])
{
    float    Modulus, Angle;
    float    currentDistance = startDistance;
    int rayPoints = (int)(rayParams.maxSamples * rayParams.samplingInterval);
    int    recordedPoints = 0;
    int sampleCount = 0;
    const Vector5 c = { 0, 0, 0, 0, 0 };                            // 5D vector for ray point coordinates

    // Determine orbit value for the starting point
    bool externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayParams.bailout, rayPoint, c);

    // Record this point as the first sample
    externalPoints[recordedPoints] = externalPoint;
    modulusValues[recordedPoints] = Modulus;
    angles[recordedPoints] = Angle;
    distances[recordedPoints] = currentDistance;
    recordedPoints++;

    // Begin loop
    while (recordedPoints < rayPoints && sampleCount < rayParams.maxSamples)
    {
        // Move on to the next point
        currentDistance += rayParams.samplingInterval;
        sampleCount++;

        // Determine orbit properties for this point
        externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayParams.bailout, rayPoint, c);

        // If this is an internal point and previous point is external
        if (rayParams.activeIndex == 0 && !externalPoint && externalPoints[recordedPoints - 1] == 1)
        {
            ///// Set value for surface point /////

            // Perform binary search between this and the previous point, to determine surface position
            float sampleDistance = FindSurface(
                rayParams.samplingInterval, rayParams.surfaceSmoothing, rayParams.binarySearchSteps,
                currentDistance, rayPoint, rayParams.bailout);
            //float sampleDistance = currentDistance;

            bool foundGap = gapFound(sampleDistance, rayParams.surfaceThickness, rayPoint, rayParams.bailout, c);

            // Test point a short distance further along, to determine whether this is still in the set
            if (rayParams.surfaceThickness > 0 && foundGap) {
                // Back outside the set, so continue as normal for external points
                externalPoint = true;
                continue;
            }
            // Determine orbit properties for this point
            externalPoint = SamplePoint(sampleDistance, &Modulus, &Angle, rayParams.bailout, rayPoint, c);

            // Save this point value in the ray collection
            externalPoints[recordedPoints] = externalPoint ? 1 : 0;
            modulusValues[recordedPoints] = Modulus;
            angles[recordedPoints] = Angle;
            distances[recordedPoints] = sampleDistance;
            recordedPoints++;
        }
        else if (rayParams.activeIndex == 1)
        {
            ///// Set value for external point /////

            const float angleChange = fabs(Angle - angles[recordedPoints - 1]);

            // If orbit value is sufficiently different from the last recorded sample
            if (angleChange > rayParams.boundaryInterval)
            {
                // Perform binary search between this and the recorded point, to determine boundary position
                float sampleDistance = FindBoundary(rayParams.samplingInterval, rayParams.binarySearchSteps, currentDistance, angles[recordedPoints - 1],
                    rayParams.boundaryInterval, &externalPoint, &Modulus, &Angle,
                    rayPoint, rayParams.bailout);

                // Save this point value in the ray collection
                externalPoints[recordedPoints] = externalPoint ? 1 : 0;
                modulusValues[recordedPoints] = Modulus;
                angles[recordedPoints] = Angle;
                distances[recordedPoints] = sampleDistance;
                recordedPoints++;
            }
        }
    }

    distances[recordedPoints] = HUGE_VAL;
    return recordedPoints + 1;
}

EXPORT float __stdcall FindSurface(
    float samplingInterval, float surfaceSmoothing, int binarySearchSteps, float currentDistance, 
    Vector3 rayPoint, float bailout)
{
    float stepFactor = surfaceSmoothing / 10;
    float    stepSize = -samplingInterval * stepFactor;
    float    sampleDistance = currentDistance;
    const Vector5 c = {0,0,0,0,0};        // 5D vector for ray point coordinates


    // Perform binary search between the current and previous points, to determine boundary position
    for (int i = 0; i < binarySearchSteps; i++)
    {
        // Step back or forwards by half the distance
        sampleDistance = sampleDistance + stepSize;

        // If this point is internal to the set
        if (!SamplePoint(sampleDistance, bailout, rayPoint, c))
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

EXPORT float __stdcall FindBoundary(float samplingInterval, int binarySearchSteps, float currentDistance, float previousAngle,
    float boundaryInterval, bool *externalPoint, float *Modulus, float *Angle, Vector3 rayPoint, float bailout)
{
    float stepSize = -samplingInterval / 2;
    float sampleDistance = currentDistance;
    const Vector5 c = {0,0,0,0,0};            // 5D vector for ray point coordinates

    // Perform binary search between the current and previous points, to determine boundary position
    for (int i = 0; i < binarySearchSteps; i++)
    {
        // Step back or forwards by half the distance
        sampleDistance = sampleDistance + stepSize;
        // Take a sample at this point
        *externalPoint = SamplePoint(sampleDistance, Modulus, Angle, bailout, rayPoint, c);

        const float angleChange = fabs(*Angle - previousAngle);

        // If this point is sufficiently different from the last recorded sample
        if (angleChange > boundaryInterval)
        {
            // Step back next time
            stepSize = -fabs(stepSize) / 2;
        }
        else
        {
            // Step forward next time
            stepSize = fabs(stepSize) / 2;
        }
    }
    return sampleDistance;
}

bool SamplePoint(float distance, float* Modulus, float* Angle, float bailout, Vector3 rayPoint, Vector5 c)
{
    // Determine the x,y,z coord for this point
    const Vector3 testPoint = rayPoint * distance;

    // Transform 3D point x,y,z into nD fractal space at point c[]
    VectorTrans(testPoint, &c);

    // Determine orbit value for this point
    return ProcessPoint(Modulus, Angle, bailout, c) ? 1 : 0;
}

// Transform 3D coordinates to 5D point c[] in fractal
// Returns true if point is external to the set
EXPORT bool __stdcall SamplePoint(float distance, float bailout, Vector3 rayPoint, Vector5 c)
{
  // Determine the x,y,z coord for this point
  const Vector3 testPoint= rayPoint * distance;

  // Transform 3D point x,y,z into nD fractal space at point c[]
  VectorTrans(testPoint, &c);

  // Determine orbit value for this point
  return ExternalPoint(c, bailout) ? 1 : 0;
}

// Determine whether nD point c[] in within the set
// Returns true if point is external to the set
bool ExternalPoint(Vector5 c, float bailout)
{
    const long MaxCount = (long)(MAX_COLOURS);          // Iteration count for external points
    Vector5 z;                                          // Temporary 5-D vector
    float ModulusTotal = 0;
    long count;

    z = c;

    for (count = 0; count < MaxCount; count++) {
        v_mandel(z.m, c.m);                           // z = z*z + c

        // Determine modulus for this point in orbit
        // Current orbit size = mod(z - c)
        Vector5 diff = z - c;
        const float ModVal = diff.dot(diff);

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

// Determine orbital modulus at nD point c[] in fractal
// Returns true if point is external to the set
bool ProcessPoint(float *Modulus, float *Angle, float bailout, Vector5 c)
{
    float const PI = 3.1415926536f;

    const int MaxCount = 100;               // Iteration count for external points
    Vector5 z;                              // Temporary 5-D vector
    float ModulusTotal = 0;
    float AngleTotal = PI;                  // Angle for first two vectors is 180 degrees
    long count;

    z = c;
    Vector5 vectorSet[3];                   // Collection of the three most recent vectors for determining the angle between them
    vectorSet[1] = z;                       // Store the first point in the vector set

    for (count = 0; count < MaxCount; count++)
    {
        v_mandel(z.m, c.m);                 // z = z * z + c
        vectorSet[2] = z;                   // Store the new point in the vector set

        // Determine vector angle for the last three positions
        if (count > 0 && count < 10) {
            // Add angle to current total
            AngleTotal += vectorAngle(vectorSet[0], vectorSet[1], vectorSet[2]);
        }

        // Determine modulus for this point in orbit
        // Current orbit size = mod(z - c)
        Vector5 diff = z - c;
        const float ModVal = diff.dot(diff);

        // Accumulate modulus value
        ModulusTotal += ModVal;

        // Stop accumulating values when modulus exceeds bailout value
        if (ModVal > bailout * bailout)
        {
            count++;
            break;
        }

        // Move the vectors down in the list for the next angle
        vectorSet[0] = vectorSet[1];
        vectorSet[1] = vectorSet[2];
    }

    // Calculate the average modulus over the orbit
    *Modulus = (float)(ModulusTotal / count);
    // Calculate the average angle over the orbit
    *Angle = (float)(AngleTotal / (count > 10 ? 10 : count + 1));

    // Return true if this point is external to the set
    return (count < MaxCount);
}

bool gapFound(float currentDistance, float surfaceThickness, Vector3 rayPoint, float bailout, Vector5 c)
{
    float testDistance;

    for (int factor = 1; factor <= 4; factor++)
    {
        testDistance = currentDistance + surfaceThickness * factor / 4;

        if (SamplePoint(testDistance, bailout, rayPoint, c))
        {
            return true;
        }
    }
    return false;
}

void VectorTrans(Vector3 imagePoint, Vector5 *c)
{
    for (int col = 0; col < DimTotal; col++)
    {
        (*c).m[col] =
            m_Trans[0][col]* imagePoint.X +          // Transforms 3D image space at point x,y,z
            m_Trans[1][col]* imagePoint.Y +          // into nD vector space at point c[]
            m_Trans[2][col]* imagePoint.Z +
            m_Trans[5][col];
    }
}

Vector5 ImageToFractalSpace(float distance, Vector3 coord)
{
    // Determine the x,y,z coord for this point
    const Vector3 rayPoint = coord * distance;

    Vector5 c = { 0,0,0,0,0 };

    // Transform 3D point x,y,z into nD fractal space at point c[]
    VectorTrans(rayPoint, &c);

    // Return the 5D fractal space point
    return c;
}
