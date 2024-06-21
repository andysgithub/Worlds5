// TRANSFORM.CPP
//
// Author:          Andrew G Williams
// Originated:      13-April-1997
// Last update:     12-February-1998
//
// Compiler:        Visual C++
// System:          PC/Windows 95-XP
//
// This is a module to perform pre-multiply and post-multiply operations
// between the manipulation and transformation matrices. This concatenates
// the matrix for the current manipulation with the overall transformation
// matrix, to produce the new transformation. The resulting matrix is then
// applied to the image plane coordinates within the 'VectorTrans' routine
// in the main module. This positions and scales the image plane in the
// 5-dimensional vector space to intersect the fractal in the chosen region.

#include "stdafx.h"
#include <math.h>
#include "unmanaged.h"

//#define trans(a,b) m_Trans[b][a]        // Macro to address transformation matrix

double manip[MAX_DIM+1][MAX_DIM+1];     // Manipulation matrix for fractal generation
int DimTotal;

/*******************************
        Module functions
 *******************************/

void ManipInit()                         // Initialise manipulation matrix
{
    int row,col;

    for (row=0; row<=MAX_DIM; ++row)
    {
        for (col=0; col<=MAX_DIM; ++col)
            manip[row][col] = 0;
        manip[row][row] = 1;
    }
}

void TransInit()                        // Initialise transformation matrix
{
    int row,col;

    for (row=0; row<=DimTotal; ++row)
    {
        for (col=0; col<=DimTotal; ++col)
            trans(row,col) = 0;
        trans(row,row) = 1;
    }
}

void PostMul()                          // Matrix post-multiply (general)
{
    int row,col,count;
    double temp[MAX_DIM+1][MAX_DIM+1];

    for (row=0;row<=DimTotal;++row)
    {
        for (col=0;col<=DimTotal;++col)
        {
            temp[row][col]=0;
            for (count=0;count<=DimTotal;++count)
                temp[row][col] += trans(row,count)*manip[count][col];
        }
    }
    for (row=0;row<=DimTotal;++row)
    {
        for (col=0;col<=DimTotal;++col)
            trans(row,col) = temp[row][col];
    }
}

void PostMulS(double Scale)            // Matrix post-multiply for scalings
{
    int row,col;

    for (row=0;row<DimTotal;++row)
    {
        for (col=0;col<DimTotal;++col)
            trans(row,col) *= Scale;
    }
}

void PreMulR()                        // Matrix pre-multiply for rotations
{
    int row,col,count;
    double temp[MAX_DIM+1][MAX_DIM+1];

    for (row=0;row<DimTotal;++row)
    {
        for (col=0;col<DimTotal;++col)
        {
            temp[row][col]=0;
            for (count=0;count<DimTotal;++count)
                temp[row][col] += manip[row][count]*trans(count,col);
        }
    }
    for (row=0;row<DimTotal;++row)
    {
        for (col=0;col<DimTotal;++col)
            trans(row,col) = temp[row][col];
    }
}

void PreMulT()                        // Matrix pre-multiply for translations
{
    int row,col;

    for (col=0;col<DimTotal;++col)
    {
        for (row=0;row<DimTotal;++row)
            trans(DimTotal,col) += manip[DimTotal][row] * trans(row,col);
    }
}

void SetRot(int Axis1,int Axis2,double Angle)
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

    //Angle = Angle/57.2957795131;

    manip[a][a] = cos(Angle);
    manip[b][b] = cos(Angle);
    manip[b][a] = sin(Angle);
    manip[a][b] = -sin(Angle);

    if (((b-a) == 2) || ((b-a) == 4))
    {
        manip[b][a] = -manip[b][a];
        manip[a][b] = -manip[a][b];
    }
}

/*********************************************************************
    Resize the transformation matrix to the new dimensions defined
    by DimCount. This is only called if the matrices are made larger
    than their current size.
 *********************************************************************/
void Redimension(int DimCount, double Scale)
{
    int row,col;

    // Copy the translation row to the new position
    for (col = 0; col < DimTotal; col++)
        trans(DimCount, col) = trans(DimTotal, col);

    // Re-initialise the rows & columns
    for (row = DimTotal; row < DimCount; row++)
    {
        for (col=0; col<=DimCount; col++)
        {
            trans(row, col) = 0;
            trans(col, row) = 0;
        }
        trans(row, row) = Scale;
    }

    // Record the new dimension count as the current value
    DimTotal = DimCount;
}
