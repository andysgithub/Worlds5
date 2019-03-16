#include "stdafx.h"
#include <math.h>
#include "unmanaged.h"
#include "declares.h"

double	m_Trans[6][6],      // Transformation to position the sphere within the 5D fractal
		m_Resolution,           // Angular resolution of the sphere surface (degrees)
		m_Radius,               // Distance from centre to first ray tracing point
		m_Latitude,							// Latitude of the viewing centre (degrees)
		m_Longitude,						// Longitude of the viewing centre (degrees)
    m_verticalView,         // Vertical field of view (degrees)
    m_horizontalView;       // Horizontal field of view (degrees)

    // Transformed reference points
double leftEdge;            // Left edge of the viewing window as seen from the sphere centre
double rightEdge;           // Right edge of the viewing window
double topEdge;             // Top edge of the viewing window
double bottomEdge;          // Bottom edge of the viewing window

float	m_Detail0, m_Detail1,
		m_Offset0, m_Offset1,
		m_Hue0, m_Saturation0, m_Lightness0,
		m_Hue1, m_Saturation1, m_Lightness1,
		m_Bailout;

bool APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved)
{
    return TRUE;
}

EXPORT void __stdcall InitSphere(float fDetail0, float fDetail1,
								  float fBailout, double dResolution,
								  float fHue0, float fSaturation0, float fLightness0,
								  float fHue1, float fSaturation1, float fLightness1,
								  double dLatitude, double dLongitude, 
								  double sphereRadius, double verticalView, double horizontalView, 
                  double *pTransMatrix)
{
	m_Detail0		= fDetail0;
	m_Detail1		= fDetail1;
	m_Bailout		= fBailout*fBailout;
	m_Resolution	= dResolution;
	m_Hue0			= fHue0;
	m_Saturation0	= fSaturation0;
	m_Lightness0	= fLightness0;
	m_Hue1			= fHue1;
	m_Saturation1	= fSaturation1;
	m_Lightness1	= fLightness1;
	m_Latitude		= dLatitude;
	m_Longitude		= dLongitude;
	m_Radius		= sphereRadius;

  // Determine viewport edge positions
  leftEdge = horizontalView / 2;
  rightEdge = -horizontalView / 2;
  topEdge = verticalView / 2;
  bottomEdge = -verticalView / 2;

  // TODO: Transform the sphere centre to the fractal space

  // Copy the supplied transformation matrix into this class
	for (int iYCount = 0; iYCount < 6; iYCount++)
	{
		for (int iXCount = 0; iXCount < 5; iXCount++)
		{
			m_Trans[iXCount][iYCount] = *(pTransMatrix + iXCount + iYCount * 5);
		}
	}
}

EXPORT void __stdcall SetViewingAngle(double dLatitude, double dLongitude)
{
	m_Latitude = dLatitude;
	m_Longitude = dLongitude;
}
