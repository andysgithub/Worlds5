#include "stdafx.h"
#include <math.h>
#include "vectors.h"

// Subtract vector A from vector B
//vector5Double vectorSub(vector5Double A, vector5Double B)
//{
//	vector5Double Result;
//
//	Result.coords[0] = B.coords[0] - A.coords[0];
//	Result.coords[1] = B.coords[1] - A.coords[1];
//	Result.coords[2] = B.coords[2] - A.coords[2];
//	Result.coords[3] = B.coords[3] - A.coords[3];
//	Result.coords[4] = B.coords[4] - A.coords[4];
//
//	return Result;
//}

//// Return the dot product of v1 and v2
//double dot(vector5Double v1, vector5Double v2)
//{
//	return v1.coords[0]*v2.coords[0] + v1.coords[1]*v2.coords[1] +
//		   v1.coords[2]*v2.coords[2] + v1.coords[3]*v2.coords[3] + 
//		   v1.coords[4]*v2.coords[4];
//}
//
//// Return the modulus of vector v
//double mod(vector5Double v)
//{
//	return sqrt(dot(v, v));
//}

//// Return the normalised vector of v
//vector5Double norm(vector5Double v)
//{
//	vector5Double v1;
//	double factor = 1 / mod(v);
//
//	v1.coords[0] = v.coords[0] * factor;
//	v1.coords[1] = v.coords[1] * factor;
//	v1.coords[2] = v.coords[2] * factor;
//	v1.coords[3] = v.coords[3] * factor;
//	v1.coords[4] = v.coords[4] * factor;
//
//	return v1;
//}

// Return angle between lines defined by vectors BA and BC
double vectorAngle(vector5Double A, vector5Double B, vector5Double C)
{
	vector5Double v1, v2;
	double dotProduct = 0;

	// Vector v1 = B - A 
	v1.coords[0] = B.coords[0] - A.coords[0];
	v1.coords[1] = B.coords[1] - A.coords[1];
	v1.coords[2] = B.coords[2] - A.coords[2];
	v1.coords[3] = B.coords[3] - A.coords[3];
	v1.coords[4] = B.coords[4] - A.coords[4];

	double modulus = sqrt(v1.coords[0]*v1.coords[0] + v1.coords[1]*v1.coords[1] +
							v1.coords[2]*v1.coords[2] + v1.coords[3]*v1.coords[3] + 
							v1.coords[4]*v1.coords[4]);

	if (modulus != 0)
	{
		double factor = 1 / modulus;

		// Normalise v1 by dividing by mod(v1)
		v1.coords[0] = v1.coords[0] * factor;
		v1.coords[1] = v1.coords[1] * factor;
		v1.coords[2] = v1.coords[2] * factor;
		v1.coords[3] = v1.coords[3] * factor;
		v1.coords[4] = v1.coords[4] * factor;

		// Vector v2 = B - C 
		v2.coords[0] = B.coords[0] - C.coords[0];
		v2.coords[1] = B.coords[1] - C.coords[1];
		v2.coords[2] = B.coords[2] - C.coords[2];
		v2.coords[3] = B.coords[3] - C.coords[3];
		v2.coords[4] = B.coords[4] - C.coords[4];

		modulus = sqrt(v2.coords[0]*v2.coords[0] + v2.coords[1]*v2.coords[1] +
							v2.coords[2]*v2.coords[2] + v2.coords[3]*v2.coords[3] + 
							v2.coords[4]*v2.coords[4]);

		if (modulus != 0)
		{
			factor = 1 / modulus;

			// Normalise v2 by dividing by mod(v2)
			v2.coords[0] = v2.coords[0] * factor;
			v2.coords[1] = v2.coords[1] * factor;
			v2.coords[2] = v2.coords[2] * factor;
			v2.coords[3] = v2.coords[3] * factor;
			v2.coords[4] = v2.coords[4] * factor;

			// Calculate dot product of v1 and v2
			dotProduct = v1.coords[0]*v2.coords[0] + v1.coords[1]*v2.coords[1] +
				   v1.coords[2]*v2.coords[2] + v1.coords[3]*v2.coords[3] + 
				   v1.coords[4]*v2.coords[4];
		}
	}

	if (dotProduct > 1)
	{
		dotProduct = 1;
	}
	else if (dotProduct < -1)
	{
		dotProduct = -1;
	}

	// Return the angle in radians
	return acos(dotProduct);

	//// Return the angle in degrees
	//return acos(dotProduct) * 57.2957795;
}
