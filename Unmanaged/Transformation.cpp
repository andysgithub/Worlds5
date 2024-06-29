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

float manip[DimTotal+1][DimTotal];     // Manipulation matrix for fractal generation

/*******************************
        Module functions
 *******************************/

void ManipInit()                         // Initialise manipulation matrix
{
    for (int row=0; row < DimTotal; ++row) {
        for (int col = 0; col < DimTotal; ++col) {
            manip[row][col] = 0;
        }
        manip[row][row] = 1;
    }
    for (int col = 0; col < DimTotal; ++col) {
        manip[DimTotal][col] = 0;
    }
}

void TransInit()                        // Initialise transformation matrix
{
    for (int row = 0; row < DimTotal; ++row)
    {
        for (int col = 0; col < DimTotal; ++col) {
            m_Trans[row][col] = 0;
        }
        m_Trans[row][row] = 1;
    }
    for (int col = 0; col < DimTotal; ++col) {
        m_Trans[DimTotal][col] = 0;
    }
}

void PostMul()                          // Matrix post-multiply (general)
{
    float result[6][5] = { 0 };

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                result[i][j] += m_Trans[i][k] * manip[k][j];
            }
        }
    }

    // Copy the result back to m_Trans
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 5; j++) {
            m_Trans[i][j] = result[i][j];
        }
    }
}

void PostMulS(float Scale)            // Matrix post-multiply for scalings
{
    for (int row = 0; row < DimTotal; ++row) {
        for (int col = 0; col < DimTotal; ++col) {
            m_Trans[row][col] *= Scale;
        }
    }
}

void PreMulR()                        // Matrix pre-multiply for rotations
{
    float temp[DimTotal][DimTotal - 1] = { 0 };

    for (int row = 0; row <= DimTotal; row++) {
        for (int col=0; col < DimTotal; col++) {
            for (int count = 0; count < DimTotal; count++) {
                temp[row][col] += m_Trans[row][count] * manip[count][col];
            }
        }
    }
    for (int row = 0; row <= DimTotal; row++) {
        for (int col = 0; col < DimTotal; col++) {
            m_Trans[row][col] = temp[row][col];
        }
    }
}

void PreMulT()                        // Matrix pre-multiply for translations
{
    for (int row = 0; row < DimTotal; row++) {
        for (int col = 0; col < DimTotal; col++) {
            m_Trans[DimTotal][col] += manip[DimTotal][row] * m_Trans[row][col];
        }
    }
}

void SetRot(int Axis1,int Axis2,float Angle)
{
    int a, b;

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

    manip[a][a] = static_cast<float>(cos(Angle));
    manip[b][b] = static_cast<float>(cos(Angle));
    manip[b][a] = static_cast<float>(sin(Angle));
    manip[a][b] = static_cast<float>(-sin(Angle));

    if (((b-a) == 2) || ((b-a) == 4))
    {
        manip[b][a] = -manip[b][a];
        manip[a][b] = -manip[a][b];
    }
}
