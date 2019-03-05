// TRANSFORM2.CPP
//
// Author:   		Andrew G Williams
// Originated:		13-April-1997
// Last update:  	12-February-1998
//
// Compiler:		Visual C++
// System:			PC/Windows 95-XP
//
// This is a module to perform pre-multiply and post-multiply operations
// between the manipulation and transformation matrices. This concatenates
// the matrix for the current manipulation with the overall transformation
// matrix, to produce the new transformation. The resulting matrix is then
// applied to the image plane coordinates within the 'VectorTrans' routine
// in the main module. This positions and scales the image plane in the
// n-dimensional vector space to intersect the fractal in the chosen region.

#include "stdafx.h"
#include <math.h>
#include "unmanaged.h"

#define trans(a,b) m_Trans2[b][a]				// Macro to address transformation matrix

extern double Scale;
double manip2[4][4];						// Manipulation matrix for image viewing
extern int DimTotal;

/*******************************
		Module functions
 *******************************/

void ManipInit2()									//	Initialise manipulation matrix
{
	int row,col;

	for (row=0; row<=MAX_DIM; ++row)
	{
		for (col=0; col<=MAX_DIM; ++col)
			manip2[row][col] = 0;
		manip2[row][row] = 1;
	}
}

void TransInit2()						//	Initialise transformation matrix
{
	int row, col;

	for (row=0; row <= DimTotal; ++row)
	{
		for (col = 0; col <= DimTotal; ++col)
			trans(row,col) = 0;
		trans(row,row) = 1;
	}
}

void PostMul2()						//	Matrix post-multiply (general)
{
	int row, col, count;
	double temp[MAX_DIM+1][MAX_DIM+1];

	for (row=0; row <= DimTotal; ++row)
	{
		for (col = 0; col <= DimTotal; ++col)
		{
			temp[row][col]=0;
			for (count=0; count <= DimTotal; ++count)
				temp[row][col] += trans(row,count) * manip2[count][col];
		}
	}
	for (row=0; row <= DimTotal; ++row)
	{
		for (col=0; col <= DimTotal; ++col)
			trans(row,col) = temp[row][col];
	}
}

void PostMulS2(double Scale)			// Matrix post-multiply for scalings
{
	int row,col;

	for (row=0; row < DimTotal; ++row)
	{
		for (col = 0; col < DimTotal; ++col)
			trans(row,col) *= Scale;
	}
}

void PreMulR2()						// Matrix pre-multiply for rotations
{
	int row, col, count;
	double temp[MAX_DIM+1][MAX_DIM+1];

	for (row = 0; row < DimTotal; ++row)
	{
		for (col = 0; col < DimTotal; ++col)
		{
			temp[row][col]=0;
			for (count = 0; count < DimTotal; ++count)
				temp[row][col] += manip2[row][count] * trans(count,col);
		}
	}
	for (row=0; row < DimTotal; ++row)
	{
		for (col=0; col < DimTotal; ++col)
			trans(row,col) = temp[row][col];
	}
}

void PreMulT2()						// Matrix pre-multiply for translations
{
	int row,col;

	for (col = 0;col < DimTotal; ++col)
	{
		for (row = 0; row < DimTotal; ++row)
			trans(DimTotal,col) += manip2[DimTotal][row] * trans(row,col);
	}
}

void SetRot2(int Axis1,int Axis2,double Angle)
{
	int a,b;

	if (Axis1 < Axis2)
	{
		a = Axis1;
		b = Axis2;
	}
	else
	{
		a = Axis2;
		b = Axis1;
	}

	Angle = Angle/57.2957795131;

	manip2[a][a] = cos(Angle);
	manip2[b][b] = cos(Angle);
	manip2[b][a] = sin(Angle);
	manip2[a][b] = -sin(Angle);

	if (((b-a) == 2) || ((b-a) == 4))
	{
		manip2[b][a] = -manip2[b][a];
		manip2[a][b] = -manip2[a][b];
	}
}


