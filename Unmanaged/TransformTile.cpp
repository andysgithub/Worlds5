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

double manipTile[4][4];						// Manipulation matrix for image viewing

/*******************************
		Module functions
 *******************************/

//	Initialise manipulation matrix
void ManipInitTile()						
{
	int row,col;

	for (row = 0; row < 4; ++row)
	{
		for (col = 0; col < 4; ++col)
			manipTile[row][col] = 0;

		manipTile[row][row] = 1;
	}
}

//	Initialise transformation matrix
void TransInitTile()						
{
	int row, col;

	for (row=0; row < 4; ++row)
	{
		for (col = 0; col < 4; ++col)
			m_TransTile[col][row] = 0;

		m_TransTile[row][row] = 1;
	}
}

//	Matrix post-multiply
void PostMulTile()						
{
	int row, col, count;
	double temp[3][3];

	for (row=0; row < 3; ++row)
	{
		for (col = 0; col < 3; ++col)
		{
			temp[row][col]=0;
			for (count = 0; count < 3; ++count)
				temp[row][col] += m_TransTile[count][row] * manipTile[count][col];
		}
	}
	for (row=0; row < 3; ++row)
	{
		for (col = 0; col < 3; ++col)
			m_TransTile[col][row] = temp[row][col];
	}
}

void SetRotTile(int Axis1, int Axis2, double Angle)
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

	manipTile[a][a] = cos(Angle);
	manipTile[b][b] = cos(Angle);
	manipTile[b][a] = sin(Angle);
	manipTile[a][b] = -sin(Angle);

	if (((b-a) == 2) || ((b-a) == 4))
	{
		manipTile[b][a] = -manipTile[b][a];
		manipTile[a][b] = -manipTile[a][b];
	}
}


