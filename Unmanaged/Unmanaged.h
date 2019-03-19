#ifndef WORLDS5_UNMANAGED_H
#define WORLDS5_UNMANAGED_H

#define EXPORT extern "C" __declspec(dllexport)

#include <windows.h>
#include "declares.h"
#include "Vectors.h"

///////  CONSTANTS  ///////
const int MAX_DIM = 20;

//////  SPHERE  //////
extern int
		m_BitmapWidth,
	    m_BitmapHeight;

extern double	
		m_Trans[6][6],
		m_Resolution,
		m_ImageRatio,
		m_Latitude,
		m_Longitude,
		m_Radius;

extern float	
		m_Detail0, m_Detail1,
		m_Offset0, m_Offset1,
		m_Hue0, m_Saturation0, m_Lightness0,
		m_Hue1, m_Saturation1, m_Lightness1,
		m_Bailout;

extern RectangleF 
		m_ImagePlane;

//////  SPHERE  //////
EXPORT	void __stdcall InitSphere(float fDetail0, float fDetail1,
										  float fBailout, double dResolution,
										  double dLatitude, double dLongitude,
										  double Radius, double verticalView, double horizontalView, double *pTransMatrix);
EXPORT	void __stdcall SetViewingAngle(double dLatitude, double dLongitude);

//////  TRANSFORMATION  //////
void	ManipInit(void);				//	Initialise manipulation matrix
void	TransInit(void);				//	Initialise transformation matrix
void	PostMul(void);  				//	Matrix post-multiply (general)
void	PostMulS(double); 				//	Matrix post-multiply for scalings
void	PreMulR(void);  				//	Matrix pre-multiply for rotations
void	PreMulT(void);  				//	Matrix pre-multiply for translations
void	SetRot(int,int,double);

//////  RAY TRACER  //////
//EXPORT void __stdcall TraceSurface(double startDistance, double increment, double surfaceThickness,
//								   double XFactor, double YFactor, double ZFactor,
//								   float *modulus, float *angle, double *distance,
//								   int rayPoints, int maxSamples, int binarySearchSteps);

EXPORT void __stdcall TraceRay(double startDistance, double increment, double surfaceThickness, 
								   double XFactor, double YFactor, double ZFactor,
                                   bool *externalsArray, float *valuesArray, float *anglesArray, double *distancesArray,
							       int rayPoints, int maxSamples, double boundaryInterval, int binarySearchSteps,
								   int activeIndex);

EXPORT double __stdcall FindSurface(double increment, int binarySearchSteps, double currentDistance,
									double xFactor, double yFactor, double zFactor);
EXPORT double __stdcall FindBoundary(double increment, int binarySearchSteps, double currentDistance, float previousAngle,
									 double boundaryInterval, bool *externalPoint, float *Modulus, float *Angle,
									 double xFactor, double yFactor, double zFactor);
EXPORT bool __stdcall ExternalPoint(vector5Double c);
EXPORT bool __stdcall ProcessPoint(float *Modulus, float *Angle, vector5Double c);
bool	gapFound(double currentDistance, double surfaceThickness, double xFactor, double yFactor, double zFactor, vector5Double c);
double	findBoundary(double increment, int binarySearchSteps, double distance, float previousAngle,
					 double boundaryInterval, bool *externalPoint, float *Modulus, float *Angle,
					 double xFactor, double yFactor, double zFactor);
bool	SamplePoint(double distance, double xFactor, double yFactor, double zFactor, vector5Double c);
bool	SamplePoint(double distance, float *Modulus, float *Angle, double xFactor, double yFactor, double zFactor, vector5Double c);
void	VectorTrans(double x, double y, double z, vector5Double *c);
void	SetLine(double xLeft, double xRight, double zLeft, double zRight, double yLeft, double yRight);
//vertex	VertexTrans(double x, double y, double z);
void	MatRotate(int Axis1,int Axis2,double Angle);

//////  COLOUR  //////
EXPORT void __stdcall HSVtoRGB(float h, float s, float v, BYTE *rval, BYTE *gval, BYTE *bval);
float	Maximum(float r, float g, float b);
float	Minimum(float r, float g, float b);
float	Value(float n1, float n2, float hue);

#endif
