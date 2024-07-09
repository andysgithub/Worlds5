#include "stdafx.h"
#include <math.h>
#include "RayProcessing.h"
#include "declares.h"
#include "kernel.cuh"

float  m_Trans[DimTotal + 1][DimTotal];

bool APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved)
{
    return TRUE;
}

EXPORT void __stdcall InitSphere(float *pTransMatrix)
{
  // Copy the supplied transformation matrix into this class
    for (int row = 0; row <= DimTotal; row++)
    {
        for (int col = 0; col < DimTotal; col++)
        {
            m_Trans[row][col] = *(pTransMatrix + row * DimTotal + col);
        }
    }
}

