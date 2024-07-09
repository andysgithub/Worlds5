#pragma once

#define EXPORT extern "C" __declspec(dllexport)

#include <windows.h>
#include "declares.h"
#include "Vectors.h"
#include "vector5Single.h"
#include "cuda_interface.h"

///////  CONSTANTS  ///////
const int MAX_DIM = 20;

//////  SPHERE  //////
extern int
		m_BitmapWidth,
	    m_BitmapHeight;

extern float	
		m_Trans[6][5],
		m_Resolution,
		m_ImageRatio,
		m_Latitude,
		m_Longitude,
		m_Radius;

extern float	
		m_Offset0, m_Offset1,
		m_Hue0, m_Saturation0, m_Lightness0,
		m_Hue1, m_Saturation1, m_Lightness1,
		m_Bailout;

typedef void (*ProgressCallback)(int rayCount, int rowCount);

//////  SPHERE  //////
EXPORT	void __stdcall InitSphere(float fBailout, float dResolution,
                                  float dLatitude, float dLongitude,
                                  float SphereRadius, float verticalView, float horizontalView, float *pTransMatrix);

//////  RAY TRACER  //////
EXPORT void __stdcall ProcessRays(RayTracingParams rayParams, RenderingParams renderParams,
	int raysPerLine, int totalLines, ProgressCallback progressCallback);

EXPORT int __stdcall TraceRay(float startDistance, RayTracingParams rayParams,
	float XFactor, float YFactor, float ZFactor,
	int externalPoints[], float modulusValues[], float angles[], float distances[]);

EXPORT float __stdcall FindSurface(float samplingInterval, float surfaceSmoothing, int binarySearchSteps, float currentDistance,
	float xFactor, float yFactor, float zFactor, float bailout);
EXPORT float __stdcall FindBoundary(float samplingInterval, int binarySearchSteps, float currentDistance, float previousAngle,
	float boundaryInterval, bool *externalPoint, float *Modulus, float *Angle,
	float xFactor, float yFactor, float zFactor, float bailout);

Vector5 ImageToFractalSpace (float distance, Vector3 coord);
EXPORT bool __stdcall SamplePoint(float distance, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c);

bool ExternalPoint(vector5Single c, float bailout);
bool ProcessPoint(float *Modulus, float *Angle, float bailout, vector5Single c);
bool gapFound(float currentDistance, float surfaceThickness, float xFactor, float yFactor, float zFactor, float bailout, vector5Single c);
bool SamplePoint(float distance, float *Modulus, float *Angle, float bailout, float xFactor, float yFactor, float zFactor, vector5Single c);
void VectorTrans(float x, float y, float z, vector5Single *c);

//////  COLOUR  //////
EXPORT void __stdcall HSVtoRGB(float h, float s, float v, BYTE *rval, BYTE *gval, BYTE *bval);
