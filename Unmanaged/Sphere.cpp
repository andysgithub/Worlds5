#include "stdafx.h"
#include <math.h>
#include "unmanaged.h"
#include "declares.h"
#include "kernel.cuh"

float  m_Trans[DimTotal+1][DimTotal],
        m_Resolution,       // Angular resolution of the sphere surface (degrees)
        m_Radius,           // Distance from centre to first ray tracing point
        m_Latitude,         // Latitude of the viewing centre (degrees)
        m_Longitude,        // Longitude of the viewing centre (degrees)
        m_verticalView,     // Vertical field of view (degrees)
        m_horizontalView;   // Horizontal field of view (degrees)

// Transformed reference points
float leftEdge;            // Left edge of the viewing window as seen from the sphere centre
float rightEdge;           // Right edge of the viewing window
float topEdge;             // Top edge of the viewing window
float bottomEdge;          // Bottom edge of the viewing window

float   m_Offset0, m_Offset1,
        m_Bailout;

bool APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved)
{
    return TRUE;
}

EXPORT void __stdcall InitSphere(
    float fBailout, float dResolution,
    float dLatitude, float dLongitude, 
    float sphereRadius, float verticalView, float horizontalView, 
    float *pTransMatrix)
{
    m_Bailout        = fBailout*fBailout;
    m_Resolution     = dResolution;
    m_Latitude       = dLatitude;
    m_Longitude      = dLongitude;
    m_Radius         = sphereRadius;

  // Determine viewport edge positions
  leftEdge = horizontalView / 2;
  rightEdge = -horizontalView / 2;
  topEdge = verticalView / 2;
  bottomEdge = -verticalView / 2;

  // Copy the supplied transformation matrix into this class
    for (int row = 0; row <= DimTotal; row++)
    {
        for (int col = 0; col < DimTotal; col++)
        {
            m_Trans[row][col] = *(pTransMatrix + row * DimTotal + col);
        }
    }
}

