#pragma once

#define EXPORT extern "C" __declspec(dllexport)

#include <windows.h>
#include "TracedRay.h"
#include "Parameters.h"

struct Vector3;
struct Vector5;

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

float vectorAngle(Vector5 A, Vector5 B, Vector5 C);

typedef void (__stdcall *ProgressCallback)(int rayCount, int rowCount, RayDataTypeIntermediate* rayData);

//////  SPHERE  //////
EXPORT    void __stdcall InitSphere(float *pTransMatrix);

//////  RAY TRACER  //////
EXPORT void __stdcall ProcessRays(RayTracingParams rayParams, RenderingParams renderParams,
    int raysPerLine, int totalLines, ProgressCallback progressCallback);

int TraceRay(float startDistance, RayTracingParams rayParams,
    float XFactor, float YFactor, float ZFactor,
    int externalPoints[], float modulusValues[], float angles[], float distances[]);

EXPORT float __stdcall FindSurface(float samplingInterval, float surfaceSmoothing, int binarySearchSteps, float currentDistance,
    float xFactor, float yFactor, float zFactor, float bailout);
EXPORT float __stdcall FindBoundary(float samplingInterval, int binarySearchSteps, float currentDistance, float previousAngle,
    float boundaryInterval, bool *externalPoint, float *Modulus, float *Angle,
    float xFactor, float yFactor, float zFactor, float bailout);

EXPORT bool __stdcall SamplePoint(float distance, float bailout, float xFactor, float yFactor, float zFactor, Vector5 c);

bool ExternalPoint(Vector5 c, float bailout);
bool ProcessPoint(float *Modulus, float *Angle, float bailout, Vector5 c);
bool gapFound(float currentDistance, float surfaceThickness, float xFactor, float yFactor, float zFactor, float bailout, Vector5 c);
bool SamplePoint(float distance, float *Modulus, float *Angle, float bailout, float xFactor, float yFactor, float zFactor, Vector5 c);
void VectorTrans(float x, float y, float z, Vector5 *c);

//////  COLOUR  //////
EXPORT void __stdcall HSVtoRGB(float h, float s, float v, BYTE *rval, BYTE *gval, BYTE *bval);
