#include "stdafx.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include "myrion.h"
#include "RayProcessing.h"
#include "declares.h"
#include "vectors.h"

#include <cuda_runtime.h>
#include "cuda_interface.h"

const BYTE MAX_COLOURS = 255;

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Host function to initialize the GPU with constant parameters
EXPORT bool InitializeGPU(const RayTracingParams* rayParams, const RenderingParams* renderParams)
{
    const cudaError_t cudaStatus = InitializeRayTracingKernel(rayParams);
    const cudaError_t cudaStatus = InitializeRenderingKernel(renderParams);
    return cudaStatus == cudaSuccess;
}

// Host function to initialize the GPU with constant parameters
EXPORT bool CopyTransformationMatrix(const float* positionMatrix)
{
    const cudaError_t cudaStatus = InitializeTransformMatrix(positionMatrix);
    return cudaStatus == cudaSuccess;
}

// Produce the collection of fractal point values for the given vector
int TraceRay(float startDistance, RayTracingParams rayParams,
    float xFactor, float yFactor, float zFactor,
    int externalPoints[], float modulusValues[], float angles[], float distances[])
{
    float	Modulus, Angle;
    float	currentDistance = startDistance;
    float	sampleDistance;
    int rayPoints = (int)(rayParams.maxSamples * rayParams.samplingInterval);
    int	recordedPoints = 0;
    int sampleCount = 0;
    const vector5Single c = { 0, 0, 0, 0, 0 };							// 5D vector for ray point coordinates

    // Determine orbit value for the starting point
    bool externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayParams.bailout, xFactor, yFactor, zFactor, c);

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
        externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, rayParams.bailout, xFactor, yFactor, zFactor, c);

        // If this is an internal point and previous point is external
        if (rayParams.activeIndex == 0 && !externalPoint && externalPoints[recordedPoints - 1] == 1)
        {
            ///// Set value for surface point /////

            // Perform binary search between this and the previous point, to determine surface position
            sampleDistance = FindSurface(
                rayParams.samplingInterval, rayParams.surfaceSmoothing, rayParams.binarySearchSteps,
                currentDistance, xFactor, yFactor, zFactor, rayParams.bailout);

            bool foundGap = gapFound(sampleDistance, rayParams.surfaceThickness, xFactor, yFactor, zFactor, rayParams.bailout, c);

            // Test point a short distance further along, to determine whether this is still in the set
            if (rayParams.surfaceThickness > 0 && foundGap) {
                // Back outside the set, so continue as normal for external points
                externalPoint = true;
                continue;
            }
            // Determine orbit properties for this point
            externalPoint = SamplePoint(sampleDistance, &Modulus, &Angle, rayParams.bailout, xFactor, yFactor, zFactor, c);

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
                sampleDistance = FindBoundary(rayParams.samplingInterval, rayParams.binarySearchSteps, currentDistance, angles[recordedPoints - 1],
                    rayParams.boundaryInterval, &externalPoint, &Modulus, &Angle,
                    xFactor, yFactor, zFactor, rayParams.bailout);

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
    float xFactor, float yFactor, float zFactor, float bailout)
{
    float stepFactor = surfaceSmoothing / 10;
	float	stepSize = -samplingInterval * stepFactor;
	float	sampleDistance = currentDistance;
	const vector5Single c = {0,0,0,0,0};        // 5D vector for ray point coordinates


    // Perform binary search between the current and previous points, to determine boundary position
    for (int i = 0; i < binarySearchSteps; i++)
    {
        // Step back or forwards by half the distance
        sampleDistance = sampleDistance + stepSize;

        // If this point is internal to the set
        if (!SamplePoint(sampleDistance, bailout, xFactor, yFactor, zFactor, c))
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
                                      float boundaryInterval, bool *externalPoint, float *Modulus, float *Angle, 
                                      float xFactor, float yFactor, float zFactor, float bailout)
{
	float stepSize = -samplingInterval / 2;
	float sampleDistance = currentDistance;
	const vector5Single c = {0,0,0,0,0};			// 5D vector for ray point coordinates

    // Perform binary search between the current and previous points, to determine boundary position
    for (int i = 0; i < binarySearchSteps; i++)
    {
        // Step back or forwards by half the distance
        sampleDistance = sampleDistance + stepSize;
        // Take a sample at this point
        *externalPoint = SamplePoint(sampleDistance, Modulus, Angle, bailout, xFactor, yFactor, zFactor, c);

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

bool SamplePoint(float distance, float* Modulus, float* Angle, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c)
{
    // Determine the x,y,z coord for this point
    const float XPos = distance * xFactor;
    const float YPos = distance * yFactor;
    const float ZPos = distance * zFactor;

    // Transform 3D point x,y,z into nD fractal space at point c[]
    VectorTrans(XPos, YPos, ZPos, &c);

    // Determine orbit value for this point
    return ProcessPoint(Modulus, Angle, bailout, c) ? 1 : 0;
}

// Transform 3D coordinates to 5D point c[] in fractal
// Returns true if point is external to the set
EXPORT bool __stdcall SamplePoint(float distance, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c)
{
  // Determine the x,y,z coord for this point
  const float XPos = distance * xFactor;
  const float YPos = distance * yFactor;
  const float ZPos = distance * zFactor;

  // Transform 3D point x,y,z into nD fractal space at point c[]
  VectorTrans(XPos, YPos, ZPos, &c);

  // Determine orbit value for this point
  return ExternalPoint(c, bailout) ? 1 : 0;
}

// Determine whether nD point c[] in within the set
// Returns true if point is external to the set
bool ExternalPoint(vector5Single c, float bailout)
{
    const long MaxCount = (long)(MAX_COLOURS);		        // Iteration count for external points
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

        //    Stop accumulating values when modulus exceeds bailout value
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
bool  ProcessPoint(float *Modulus, float *Angle, float bailout, vector5Single c)
{
    float const PI = 3.1415926536f;

	const long MaxCount = (long)(100);		// Iteration count for external points
	vector5Single	z;						// Temporary 5-D vector
	vector5Single diff;						// Temporary 5-D vector for orbit size
	float ModulusTotal = 0;
	float ModVal = 0;
	float AngleTotal = PI;		// Angle for first two vectors is 180 degrees
	long count;

    z.coords[DimTotal - 2] = 0;
    z.coords[DimTotal - 1] = 0;

    v_mov(c.coords, z.coords);             // z = c
    vector5Single vectorSet[3];            // Collection of the three most recent vectors for determining the angle between them
    v_mov(z.coords, vectorSet[1].coords);  // Store the first point in the vector set

    for (count = 0; count < MaxCount; count++)
    {
        v_mandel(z.coords, c.coords);                // z = z*z + c
        v_mov(z.coords, vectorSet[2].coords);        // Store the new point in the vector set

        // Determine vector angle for the last three positions
        if (count > 0 && count < 10)
        {
            // Add angle to current total
            AngleTotal += vectorAngle(vectorSet[0], vectorSet[1], vectorSet[2]);
        }

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

        // Move the vectors down in the list for the next angle
        v_mov(vectorSet[1].coords, vectorSet[0].coords);
        v_mov(vectorSet[2].coords, vectorSet[1].coords);
    }

    // Calculate the average modulus over the orbit
    *Modulus = (float)(ModulusTotal / count);
    // Calculate the average angle over the orbit
    *Angle = (float)(AngleTotal / (count > 10 ? 10 : count + 1));

    // Return true if this point is external to the set
    return (count < MaxCount);
}

bool gapFound(float currentDistance, float surfaceThickness, float xFactor, float yFactor, float zFactor, float bailout, vector5Single c)
{
    float testDistance;

    for (int factor = 1; factor <= 4; factor++)
    {
        testDistance = currentDistance + surfaceThickness * factor / 4;

        if (SamplePoint(testDistance, bailout, xFactor, yFactor, zFactor, c))
        {
            return true;
        }
    }
    return false;
}

void VectorTrans(float x, float y, float z, vector5Single *c)
{
    for (int col = 0; col < DimTotal; col++)
    {
        (*c).coords[col] =
            m_Trans[0][col]*x +          // Transforms 3D image space at point x,y,z
            m_Trans[1][col]*y +          // into nD vector space at point c[]
            m_Trans[2][col]*z +
            m_Trans[5][col];
    }
}

Vector5 ImageToFractalSpace(float distance, Vector3 coord)
{
    // Determine the x,y,z coord for this point
    const float XPos = distance * coord.X;
    const float YPos = distance * coord.Y;
    const float ZPos = distance * coord.Z;

    vector5Single c = { 0,0,0,0,0 };

    // Transform 3D point x,y,z into nD fractal space at point c[]
    VectorTrans(XPos, YPos, ZPos, &c);

    // Return the 5D fractal space point
    std::array<float, 5> arr = c.toArray();
    return Vector5(arr[0], arr[1], arr[2], arr[3], arr[4]);
}

// Function to handle CUDA errors
void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}