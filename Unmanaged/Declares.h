#ifndef _DECLARES_H
#define _DECLARES_H

/* Type structures */

//typedef struct vertex_t
//{
//    double X;
//    double Y;
//    double Z;
//}
//vertex;

typedef struct rectangleFType
{
	float Left;
	float Top;
	float Width;
	float Height;
}
RectangleF;

//typedef struct rectangle_t
//{
//	int Left;
//	int Top;
//	int Width;
//	int Height;
//}
//RectangleI;

//typedef struct pixel_t
//{
//	BYTE Blue;
//	BYTE Green;
//	BYTE Red;
//}
//pixel;
//
//typedef struct point_t
//{
//	int X;
//	int Y;
//}
//point;

const float inv60 = (float)0.01667;

#endif
