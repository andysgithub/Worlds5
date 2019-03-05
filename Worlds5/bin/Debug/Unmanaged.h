#ifndef _UNMANAGED_H
#define _UNMANAGED_H
#define EXPORT extern "C" __declspec(dllexport)

#include <windows.h>

EXPORT void __stdcall InitParams(double *pTransMatrix, float fDetail0, float fDetail1,
										  float fBailout, float fOffset0, float fOffset1,
										  BYTE *pPalette0, BYTE *pPalette1,
										  float fHue0, float fSaturation0, float fLightness0,
										  float fHue1, float fSaturation1, float fLightness1);
EXPORT long __stdcall PColour(float *Modulus);
EXPORT float __stdcall DownloadPoint(double x, double y, double z, float *Modulus);
EXPORT void __stdcall VectorTrans(double x, double y, double z);

EXPORT void __stdcall GetColour(double XPos, double YPos, double ZPos, float *Modulus, BYTE *r, BYTE *g, BYTE *b);
EXPORT void __stdcall InterpolateRGB(float Fraction, BYTE *r, BYTE *g, BYTE *b, BYTE *Palette, float Hue, float Saturation, float Lightness);
EXPORT void __stdcall RGBtoHSV(BYTE rval, BYTE gval, BYTE bval, float *h, float *s, float *v);
EXPORT void __stdcall HSVtoRGB(float h, float s, float v, BYTE *rval, BYTE *gval, BYTE *bval);
EXPORT void __stdcall RGBtoHSL(BYTE rval, BYTE gval, BYTE bval, float *h, float *s, float *l);
EXPORT void __stdcall HSLtoRGB(float h, float s, float l, BYTE *rval, BYTE *gval, BYTE *bval, float Lightness);
float Maximum(float r, float g, float b);
float Minimum(float r, float g, float b);
float Value(float n1, float n2, float hue);

#endif
