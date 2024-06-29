#include "stdafx.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include "myrion.h"
#include "unmanaged.h"
#include "declares.h"
#include "vectors.h"
#include "vector5Single.h"

#include <cuda_runtime.h>
#include "kernel.cuh"
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
EXPORT bool InitializeGPU(const RayTracingParams* params)
{
    cudaError_t cudaStatus = InitializeGPUKernel(params);
    return cudaStatus == cudaSuccess;
}

// Host function to initialize the GPU with constant parameters
EXPORT bool CopyTransformationMatrix(const float* positionMatrix)
{
    cudaError_t cudaStatus = InitializeTransformMatrix(positionMatrix);
    return cudaStatus == cudaSuccess;
}

EXPORT bool VerifyTransformationMatrix(float* output)
{
    cudaError_t cudaStatus = VerifyTransformMatrix(output);
    return cudaStatus == cudaSuccess;
}

int TraceRayC(float startDistance, float increment, float smoothness, float surfaceThickness,
    float xFactor, float yFactor, float zFactor, float bailout,
    int externalPoints[], float modulusValues[], float angles[], float distances[],
    int rayPoints, int maxSamples, float boundaryInterval, int binarySearchSteps,
    int activeIndex)
{
    float	Modulus, Angle;
    float	currentDistance = startDistance;
    float	sampleDistance;
    int	recordedPoints = 0;
    int sampleCount = 0;
    const vector5Single c = { 0, 0, 0, 0, 0 };							// 5D vector for ray point coordinates

    // Determine orbit value for the starting point
    bool externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, bailout, xFactor, yFactor, zFactor, c);

    // Record this point as the first sample
    externalPoints[recordedPoints] = externalPoint;
    modulusValues[recordedPoints] = Modulus;
    angles[recordedPoints] = Angle;
    distances[recordedPoints] = currentDistance;
    recordedPoints++;

    // Begin loop
    while (recordedPoints < rayPoints && sampleCount < maxSamples)
    {
        // Move on to the next point
        currentDistance += increment;
        sampleCount++;

        // Determine orbit properties for this point
        externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, bailout, xFactor, yFactor, zFactor, c);

        //printf("surface point: %s\n", (activeIndex == 0 && !externalPoint && externalPoints[recordedPoints - 1] == 1) ? "true" : "false");

        // If this is an internal point and previous point is external
        if (activeIndex == 0 && !externalPoint && externalPoints[recordedPoints - 1] == 1)
        {
            ///// Set value for surface point /////

            // Perform binary search between this and the previous point, to determine surface position
            sampleDistance = FindSurface(increment, smoothness, binarySearchSteps, currentDistance, xFactor, yFactor, zFactor, bailout);

            // Test point a short distance further along, to determine whether this is still in the set
            if (surfaceThickness > 0 && gapFound(sampleDistance, surfaceThickness, xFactor, yFactor, zFactor, bailout, c))
            {
                // Back outside the set, so continue as normal for external points
                externalPoint = true;
                continue;
            }
            // Determine orbit properties for this point
            externalPoint = SamplePoint(sampleDistance, &Modulus, &Angle, bailout, xFactor, yFactor, zFactor, c);

            // Save this point value in the ray collection
            externalPoints[recordedPoints] = externalPoint ? 1 : 0;
            modulusValues[recordedPoints] = Modulus;
            angles[recordedPoints] = Angle;
            distances[recordedPoints] = sampleDistance;
            recordedPoints++;
        }
        else if (activeIndex == 1)
        {
            ///// Set value for external point /////

            const float angleChange = fabs(Angle - angles[recordedPoints - 1]);

            // If orbit value is sufficiently different from the last recorded sample
            if (angleChange > boundaryInterval)
            {
                // Perform binary search between this and the recorded point, to determine boundary position
                sampleDistance = FindBoundary(increment, binarySearchSteps, currentDistance, angles[recordedPoints - 1],
                    boundaryInterval, &externalPoint, &Modulus, &Angle,
                    xFactor, yFactor, zFactor, bailout);

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

int TraceRayCuda(float XFactor, float YFactor, float ZFactor, int rayPoints,
    int externalPoints[], float modulusValues[], float angles[], float distances[])
{
    // Allocate host memory if not already done
    // Note: In a real-world scenario, you might want to manage this memory externally for better performance
    int* h_externalPoints = externalPoints;
    float* h_modulusValues = modulusValues;
    float* h_angles = angles;
    float* h_distances = distances;

    // Call the CUDA kernel wrapper
    int recordedPoints = launchTraceRayKernel(
        XFactor, YFactor, ZFactor, rayPoints,
        h_externalPoints, h_modulusValues, h_angles, h_distances
    );

    // Check for CUDA errors
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    distances[recordedPoints] = HUGE_VAL;

    return recordedPoints + 1;
}

// Produce the collection of fractal point values for the given vector
EXPORT int __stdcall TraceRay(float startDistance, float increment, float smoothness, float surfaceThickness,
    float XFactor, float YFactor, float ZFactor, float bailout,
    int externalPoints[], float modulusValues[], float angles[], float distances[],
    int rayPoints, int maxSamples, float boundaryInterval, int binarySearchSteps,
    int activeIndex)
{
    //return TraceRayC(startDistance, increment, smoothness, surfaceThickness,
    //    XFactor, YFactor, ZFactor, bailout,
    //    externalPoints, modulusValues, angles, distances,
    //    rayPoints, maxSamples, boundaryInterval, binarySearchSteps,
    //    activeIndex);

    return TraceRayCuda(XFactor, YFactor, ZFactor, rayPoints,
        externalPoints, modulusValues, angles, distances);
}

EXPORT float __stdcall FindSurface(float increment, float smoothness, int binarySearchSteps, float currentDistance, float xFactor, float yFactor, float zFactor, float bailout)
{
    float stepFactor = smoothness / 10;
	float	stepSize = -increment * stepFactor;
	float	sampleDistance = currentDistance;
	const vector5Single c = {0,0,0,0,0};							// 5D vector for ray point coordinates


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

EXPORT float __stdcall FindBoundary(float increment, int binarySearchSteps, float currentDistance, float previousAngle,
                                      float boundaryInterval, bool *externalPoint, float *Modulus, float *Angle, 
                                      float xFactor, float yFactor, float zFactor, float bailout)
{
	float stepSize = -increment / 2;
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

EXPORT std::array<float, 5> __stdcall ImageToFractalSpace (float distance, float xFactor, float yFactor, float zFactor)
{
    // Determine the x,y,z coord for this point
    const float XPos = distance * xFactor;
    const float YPos = distance * yFactor;
    const float ZPos = distance * zFactor;

    vector5Single c = { 0,0,0,0,0 };

    // Transform 3D point x,y,z into nD fractal space at point c[]
    VectorTrans(XPos, YPos, ZPos, &c);

    // Return the nD fractal space point
    return c.toArray();
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


// Function to handle CUDA errors
void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
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
    for (int i = 0; i < DimTotal; i++)
    {
        (*c).coords[i] = m_Trans[i][0]*x +          // Transforms 3D image space at point x,y,z
               m_Trans[i][1]*y +                    // into nD vector space at point c[]
               m_Trans[i][2]*z +
               m_Trans[i][5];
    }
}

//vertex VertexTrans(float x, float y, float z)
//{
//    vertex v;
//
//    v.X = m_Trans[0][0]*x +                            // Transforms vertex at point x,y,z into new vertex
//          m_Trans[0][1]*y +
//          m_Trans[0][2]*z;
//
//    v.Y = m_Trans[1][0]*x +
//          m_Trans[1][1]*y +
//          m_Trans[1][2]*z;
//
//    v.Z = m_Trans[2][0]*x +
//          m_Trans[2][1]*y +
//          m_Trans[2][2]*z;
//
//    return v;
//}

