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

// Determine the fractal colour for this modulus value
//EXPORT void __stdcall GetColour(bool externalPoint, float Modulus, float Angle, BYTE *r, BYTE *g, BYTE *b) 
//{
//	float Lightness;
//	float Saturation;
//
//	if (externalPoint) 
//	{
//		// Set lightness and saturation according to modulus value
//		Lightness = 1;
//		Saturation = 1;
//	}
//	else
//	{
//		// Set lightness and saturation according to modulus value
//		Lightness = 1;
//		Saturation = 1;
//	}
//
//	// Set hue according to angle value
//	float Hue = (float)(Angle * 57.2957795 * 2);
//	// Convert resulting HSV to RGB
//	HSVtoRGB(Hue, Saturation, Lightness, r, g, b);
//}

//void RGBtoHSV(BYTE rval, BYTE gval, BYTE bval, float *h, float *s, float *v)
//{
///*	Inputs: r,g,b each in [0,255]
//	Outputs: h in [0,360]
//			 s in [0,1]
//			 v in [0,1]
//	If s=0 then 0 returned in h
//*/
//
//	float max, min;
//	float range;
//	float r, g, b;
//	float rc, gc, bc;
//
//	r = (float)rval / 255;
//	g = (float)gval / 255;
//	b = (float)bval / 255;
//
//	max = Maximum(r,g,b);
//	min = Minimum(r,g,b);
//
//	range = max - min;
//	*v = max;
//
//	if (max != 0)
//		*s = range/max;
//	else
//		*s = 0;
//
//	if (*s == 0)
//		*h = 0;
//	else
//	{
//		rc = (max - r)/range;
//		gc = (max - g)/range;
//		bc = (max - b)/range;
//
//		if (r == max)
//			*h = bc - gc;
//		else if (g == max)
//			*h = 2 + rc - bc;
//		else if (b == max)
//			*h = 4 + gc - rc;
//
//		*h = *h * 60;
//
//		if (*h < 0) *h += 360;
//	}
//}
//
//EXPORT void __stdcall HSVtoRGB(float h, float s, float v, BYTE *rval, BYTE *gval, BYTE *bval)
//{
// /*	Inputs:	 h in [0,360]
//			 s in [0,1]
//			 v in [0,1]
//	Outputs: r,g,b each in [0,255]
// */
//	float f, i;
//	float p, q, t;
//	float r, g, b;
//
//	if (s == 0)
//	{
//		r = v; g = v; b = v;
//	}
//	else
//	{
//		p = v * (1 - s);
//
//		if (h == 360)
//		{
//			r = v; g = p; b = p;
//		}
//		else
//		{
//			h *= inv60;
//			i = (float)floor((long double)h);	// Primary colour
//			f = h - i;						// Secondary colour
//			q = v * (1 - s*f);
//			t = v * (1 - s*(1 - f));
//
//			switch ((int)i)
//			{
//				case 0:
//					r = v; g = t; b = p;
//					break;
//				case 1:
//					r = q; g = v; b = p;
//					break;
//				case 2:
//					r = p; g = v; b = t;
//					break;
//				case 3:
//					r = p; g = q; b = v;
//					break;
//				case 4:
//					r = t; g = p; b = v;
//					break;
//				case 5:
//					r = v; g = p; b = q;
//					break;
//			}
//		}
//	}
//
//	*rval = (BYTE)(r*255);
//	*gval = (BYTE)(g*255);
//	*bval = (BYTE)(b*255);
//}

EXPORT void __stdcall HSVtoRGB(float h, float s, float v, BYTE *rval, BYTE *gval, BYTE *bval) 
{
 /*	
    Inputs:	 h in [0,360]
			 s in [0,1]
			 v in [0,1]
	Outputs: r,g,b each in [0,255]
 */
	float f, i;
	float p, q, t;
	float r, g, b;					// rgb values of 0.0 - 1.0
									// s and v are from 0.0 - 1.0)
	if (h == 360)
		h = 0;

	h *= inv60;						// convert hue to be in 0,6 
	i = (float)floor((double)h);	// i = greatest integer <= h 
	f = h - i;						// f = fractional part of h 

	p = v * (1 - s);
	q = v * (1 - (s * f));
	t = v * (1 - (s * (1 - f)));

	switch ((int) i)
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
	}

	*rval = (BYTE)(r * 255); /* Normalise the values to 255 */
	*gval = (BYTE)(g * 255);
	*bval = (BYTE)(b * 255);
	return;
}

//void RGBtoHSL(BYTE rval, BYTE gval, BYTE bval, float *h, float *s, float *l)
//{
///*	Inputs:	 rval, gval, bval each in [0,255]
//			 producing r,g,b each in [0,1]
//	Outputs: h in [0,360]
//			 l in [0,1]
//			 s in [0,1]
//	If s=0 then 0 returned in h
//*/
//
//	float max, min;
//	float sum, diff;
//	float r, g, b;
//	float rc, gc, bc;
//
//	extern float Maximum(float, float, float);
//	extern float Minimum(float, float, float);
//
//	r = (float)rval / 255;
//	g = (float)gval / 255;
//	b = (float)bval / 255;
//
//	max = Maximum(r,g,b);
//	min = Minimum(r,g,b);
//
//	sum = max + min;
//	diff = max - min;
//
//	*l = sum/2;			// Lightness
//
//	// Calculate saturation
//	if (max == min)				// Achromatic case
//	{
//		*s = 0;					
//		*h = 0;
//		return;
//	}
//	else						// Chromatic case
//	{
//		if (*l <= 0.5)
//			*s = diff/sum;
//		else
//			*s = diff/(2 - sum);
//
//		// Calculate hue
//		rc = (max - r)/diff;
//		gc = (max - g)/diff;
//		bc = (max - b)/diff;
//
//		if (r == max)
//			*h = bc - gc;
//		else if (g == max)
//			*h = 2 + rc - bc;
//		else if (b == max)
//			*h = 4 + gc - rc;
//
//		*h *= 60;				// Convert to degrees
//		if (*h < 60) *h += 360;	// Make non-negative
//	}
//}
//
//void HSLtoRGB(float h, float s, float l, BYTE *rval, BYTE *gval, BYTE *bval, float Lightness)
//{
///*	Inputs:	 h in [0,360]
//			 l in [0,1]
//			 s in [0,1]
//	Outputs: rval,gval,bval each in [0,255]
//*/
//	float m1, m2;
//	float r, g, b;
//
//	extern float Value(float, float, float);
//
//	if (h < 0) h = 0;
//	if (s < 0) s = 0;
//	if (l < 0) l = 0;
//
//	if (l <= 0.5)
//		m2 = l*(1 + s);
//	else
//		m2 = l + s - l*s;
//
//	m1 = 2*l - m2;
//
//	if (s == 0)
//		r = g = b = l;
//	else
//	{
//		r = Value(m1, m2, h + 120);
//		g = Value(m1, m2, h);
//		b = Value(m1, m2, h - 120);
//	}
//
//	l = Lightness/100;
//	if (l>0)
//	{
//		r += l*(1-r);
//		g += l*(1-g);
//		b += l*(1-b);
//	}
//	else
//	{
//		r += l*r;
//		g += l*g;
//		b += l*b;
//	}
//
//	if (r < 0) r = 0;
//	if (r > 1) r = 1;
//	if (g < 0) g = 0;
//	if (g > 1) g = 1;
//	if (b < 0) b = 0;
//	if (b > 1) b = 1;
//
//	*rval = (BYTE)(r*255);
//	*gval = (BYTE)(g*255);
//	*bval = (BYTE)(b*255);
//}
//
//float Maximum(float r, float g, float b)
//{
//	float max;
//
//	// Determine the maximum of r, g & b
//	max = g;
//	if (r > g) max = r;
//	if (b > max) max = b;
//
//	return max;
//}
//
//float Minimum(float r, float g, float b)
//{
//	float min;
//
//	// Determine the minimum of r, g & b
//	min = g;
//	if (r < g) min = r;
//	if (b < min) min = b;
//
//	return min;
//}
//
//float Value(float n1, float n2, float hue)
//{
//	if (hue > 360)	hue = hue -360;
//	if (hue < 0)	hue = hue + 360;
//
//	if (hue < 60)	return (n1 + (n2 - n1) * hue/60);
//	if (hue <180)	return n2;
//	if (hue <240)	return n1 + (n2 - n1)*(240 - hue)/60;
//
//	return n1;
//}