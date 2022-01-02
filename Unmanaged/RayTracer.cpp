#include "stdafx.h"
#include <math.h>
#include "apeirion.h"
#include "unmanaged.h"
#include "declares.h"
#include "vectors.h"

#define trans(a,b) m_Trans[b][a]            // Macro to address transformation matrix
int        DimTotal = 5;                    // Total number of dimensions used

const BYTE    MAX_COLOURS = 1000;
const BYTE    MAX_RAY_POINTS = 100;

bool gapFound(double currentDistance, double surfaceThickness, double xFactor, double yFactor, double zFactor, vector5Double c)
{
    double testDistance;

    for (int factor = 1; factor <= 4; factor++)
    {
        testDistance = currentDistance + surfaceThickness * factor / 4;

        if (SamplePoint(testDistance, xFactor, yFactor, zFactor, c) == 1)
        {
            return true;
        }
    }
    return false;
}

// Produce the collection of fractal point values for the given vector
EXPORT int __stdcall TraceRay(double startDistance, double increment, double surfaceThickness, 
                               double XFactor, double YFactor, double ZFactor,
                               int externalPoints[], float modulusValues[], float angles[], double distances[],
                               int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
                               int activeIndex)
{
	float	Modulus, Angle;
	double	currentDistance = startDistance;
	double	sampleDistance;
	int	recordedPoints = 0;
	int sampleCount = 0;
	const double xFactor = XFactor;
	const double yFactor = YFactor;
	const double zFactor = ZFactor;
	const vector5Double c = {0,0,0,0,0};							// 5D vector for ray point coordinates

    // Determine orbit value for the starting point
    bool externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, c);

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
        externalPoint = SamplePoint(currentDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, c);

        // If this is an internal point and previous point is external
        if (activeIndex == 0 && externalPoint == 0 && externalPoints[recordedPoints - 1] == 1)
        {
            ///// Set value for surface point /////

            // Perform binary search between this and the previous point, to determine surface position
            sampleDistance = FindSurface(increment, binarySearchSteps, currentDistance, xFactor, yFactor, zFactor);

            // Test point a short distance further along, to determine whether this is still in the set
            if (surfaceThickness > 0 && gapFound(sampleDistance, surfaceThickness, xFactor, yFactor, zFactor, c))
            {
                // Back outside the set, so continue as normal for external points
                externalPoint = true;
                continue;
            }
            // Determine orbit properties for this point
            externalPoint = SamplePoint(sampleDistance, &Modulus, &Angle, xFactor, yFactor, zFactor, c);

            // Save this point value in the ray collection
            externalPoints[recordedPoints] = externalPoint;
            modulusValues[recordedPoints] = Modulus;
            angles[recordedPoints] = Angle;
            distances[recordedPoints] = sampleDistance;
            recordedPoints++;
        }
        else if (activeIndex == 1)
        {
            ///// Set value for external point /////

			const double angleChange = fabs(Angle - angles[recordedPoints-1]);

            // If orbit value is sufficiently different from the last recorded sample
            if (angleChange > boundaryInterval)
            {
                // Perform binary search between this and the recorded point, to determine boundary position
                sampleDistance = FindBoundary(increment, binarySearchSteps, currentDistance, angles[recordedPoints-1],
                                                     boundaryInterval, &externalPoint, &Modulus, &Angle,
                                                     xFactor, yFactor, zFactor);

                // Save this point value in the ray collection
                externalPoints[recordedPoints] = externalPoint;
                modulusValues[recordedPoints] = Modulus;
                angles[recordedPoints] = Angle;
                distances[recordedPoints] = sampleDistance;
                recordedPoints++;
            }
        }
    }

    distances[recordedPoints] = HUGE_VAL;
    return recordedPoints+1;
}

EXPORT double __stdcall FindSurface(double increment, int binarySearchSteps, double currentDistance, double xFactor, double yFactor, double zFactor)
{
	double	stepSize = -increment / 2;
	double	sampleDistance = 0;
	const vector5Double c = {0,0,0,0,0};							// 5D vector for ray point coordinates


    // Perform binary search between the current and previous points, to determine boundary position
    for (int i = 0; i < binarySearchSteps; i++)
    {
        // Step back or forwards by half the distance
        sampleDistance = currentDistance + stepSize;

        // If this point is internal to the set
        if (SamplePoint(sampleDistance, xFactor, yFactor, zFactor, c) == 0)
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

EXPORT double __stdcall FindBoundary(double increment, int binarySearchSteps, double currentDistance, float previousAngle,
                                      double boundaryInterval, bool *externalPoint, float *Modulus, float *Angle, 
                                      double xFactor, double yFactor, double zFactor)
{
	double stepSize = -increment / 2;
	double sampleDistance = 0;
	const vector5Double c = {0,0,0,0,0};			// 5D vector for ray point coordinates

    // Perform binary search between the current and previous points, to determine boundary position
    for (int i = 0; i < binarySearchSteps; i++)
    {
        // Step back or forwards by half the distance
        sampleDistance = currentDistance + stepSize;
        // Take a sample at this point
        *externalPoint = SamplePoint(sampleDistance, Modulus, Angle, xFactor, yFactor, zFactor, c);

		const double angleChange = fabs(*Angle - previousAngle);

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

EXPORT bool __stdcall SamplePoint(double distance, double xFactor, double yFactor, double zFactor, vector5Double c)
{
  // Determine the x,y,z coord for this point
  const double XPos = distance * xFactor;
  const double YPos = distance * yFactor;
  const double ZPos = distance * zFactor;

  // Transform 3D point x,y,z into nD fractal space at point c[]
  VectorTrans(XPos, YPos, ZPos, &c);

  // Determine orbit value for this point
  return ExternalPoint(c) ? 1 : 0;
}

bool SamplePoint(double distance, float *Modulus, float *Angle, double xFactor, double yFactor, double zFactor, vector5Double c)
{
  // Determine the x,y,z coord for this point
  const double XPos = distance * xFactor;
  const double YPos = distance * yFactor;
  const double ZPos = distance * zFactor;

  // Transform 3D point x,y,z into nD fractal space at point c[]
  VectorTrans(XPos, YPos, ZPos, &c);

  // Determine orbit value for this point
  return ProcessPoint(Modulus, Angle, c) ? 1 : 0;
}

//    Determine whether nD point c[] in within the set
//  Returns true if point is external to the set
bool ExternalPoint(vector5Double c)
{
    const long MaxCount = (long)(MAX_COLOURS / m_Detail1);		// Iteration count for external points
	vector5Double	z;												// Temporary 5-D vector
	vector5Double diff;											// Temporary 5-D vector for orbit size
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
        if (ModVal > m_Bailout)
        {
            count++;
            break;
        }
    }

    // Return true if this point is external to the set
    return (count < MaxCount);
}

//    Determine orbital modulus at nD point c[] in fractal
//  Returns true if point is external to the set
bool  ProcessPoint(float *Modulus, float *Angle, vector5Double c)
{
    double const PI = 3.1415926536;
    double const PI_OVER_2 = PI/2;

	const long MaxCount = (long)(1000);		// Iteration count for external points
	vector5Double	z;												// Temporary 5-D vector
	vector5Double diff;											// Temporary 5-D vector for orbit size
	double ModulusTotal = 0;
	double ModVal = 0;
	double AngleTotal = PI;		// Angle for first two vectors is 180 degrees
	long count;

    z.coords[DimTotal - 2] = 0;
    z.coords[DimTotal - 1] = 0;

    v_mov(c.coords, z.coords);             // z = c
    vector5Double vectorSet[3];            // Collection of the three most recent vectors for determining the angle between them
    v_mov(z.coords, vectorSet[1].coords);            // Store the first point in the vector set

    for (count = 0; count < MaxCount; count++)
    {
        v_mandel(z.coords, c.coords);                //    z = z*z + c
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

        //    Stop accumulating values when modulus exceeds bailout value
        if (ModVal > m_Bailout)
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

void VectorTrans(double x, double y, double z, vector5Double *c)
{
    for (int i = 0; i < DimTotal; i++)
    {
        (*c).coords[i] = m_Trans[i][0]*x +          // Transforms 3D image space at point x,y,z
               m_Trans[i][1]*y +                    // into nD vector space at point c[]
               m_Trans[i][2]*z +
               m_Trans[i][5];
    }
}

//vertex VertexTrans(double x, double y, double z)
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
