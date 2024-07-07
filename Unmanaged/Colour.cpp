/* COLOUR.CPP
 *
 * Author:   		Andrew G Williams
 * Originated:		9-December-2000
 * Last update:		9-November-2002
 *
 * Compiler:		Visual C++ v6.0
 * System:			PC/Windows 2000
 *
 * Class name:		None
 * Public member:	None
 *
 * This module performs conversions between colour models.
 *
 */

#include "stdafx.h"
#include <math.h>
#include "declares.h"
#include "unmanaged.h"
#include "Colour.h"

/*
	Inputs:	 h in [0,360]
			 s in [0,1]
			 v in [0,1]
	Outputs: r,g,b each in [0,255]
*/
COLOUR_API void __stdcall HSVtoRGB(float h, float s, float v, unsigned char* rval, unsigned char* gval, unsigned char* bval)
{
	float f, i;
	float p, q, t;
	float r = 0, g = 0, b = 0;		// rgb values of 0.0 - 1.0
									// s and v are from 0.0 - 1.0)
	if (h == 360)
		h = 0;

	h *= inv60;						// convert hue to be in 0,6 
	i = (float)floor((float)h);	// i = greatest integer <= h 
	f = h - i;						// f = fractional part of h 

	p = v * (1 - s);
	q = v * (1 - (s * f));
	t = v * (1 - (s * (1 - f)));

	switch ((int)i)
	{
	case 0:
		r = v; g = t; b = p;
		break;
	case 1:
		r = q; g = v; b = p;
		break;
	case 2:
		r = p; g = v; b = t;
		break;
	case 3:
		r = p; g = q; b = v;
		break;
	case 4:
		r = t; g = p; b = v;
		break;
	case 5:
		r = v; g = p; b = q;
		break;
	default:
		r = 0; g = 0; b = 0;
	}

	// Normalise the values to [0,255]
	*rval = static_cast<unsigned char>(r * 255);
	*gval = static_cast<unsigned char>(g * 255);
	*bval = static_cast<unsigned char>(b * 255);
	return;
}
