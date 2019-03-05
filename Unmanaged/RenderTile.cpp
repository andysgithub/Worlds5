#include "stdafx.h"
#include <math.h>
#include "unmanaged.h"
#include "declares.h"

int m_TilePixelIndex = 0;
int m_BitmapWidth = 0;
int m_BitmapHeight = 0;
double m_ImageRatio = 0;

int	m_TilePatchNo;
BYTE* m_Scan0;
int m_Stride;

EXPORT void __stdcall InitBitmap(int Width, int Height)
{
    m_BitmapWidth = Width;
    m_BitmapHeight = Height;
}

EXPORT int __stdcall ShowPatch(int PatchNo, BYTE* Scan0, int Stride)
{
	if (PatchNo == 20479)
		PatchNo=PatchNo;

	// Don't process if patch is empty
    if (patches_PixelData[PatchNo][0].Red == 0 &&
        patches_PixelData[PatchNo][0].Green == 0 &&
        patches_PixelData[PatchNo][0].Blue == 0)
        return 1;

	m_Scan0 = Scan0;
	m_Stride = Stride;
    m_ImageRatio = (double)m_BitmapWidth / m_ImagePlane.Width;

	m_TilePatchNo = PatchNo;
    m_TilePixelIndex = 0;

	// Read the patch reference vertices
	vertex v0 = patches_Reference[PatchNo][0];
    vertex v1 = patches_Reference[PatchNo][1];
    vertex v2 = patches_Reference[PatchNo][2];

	// Order vertices by Y values
    if (v1.Y < v0.Y) SwapVertices(&v0, &v1);
    if (v2.Y < v1.Y) SwapVertices(&v1, &v2);
    if (v1.Y < v0.Y) SwapVertices(&v0, &v1);

    // Initialise the left/right coords to the first vertex
    double xleft = v0.X, xright = v0.X;
    double zleft = v0.Z, zright = v0.Z;
	double dxleft = 0, dxright = 0;
    double dzleft = 0, dzright = 0;

	double yInc = m_Resolution;

	// Initialise the increments between lines from y0 to y1
	if (v2.Y - v0.Y != 0)
	{
		dxleft = (v2.X - v0.X) / (v2.Y - v0.Y) * yInc;
		dzleft = (v2.Z - v0.Z) / (v2.Y - v0.Y) * yInc;
	}
	if (v1.Y - v0.Y != 0) 
	{
		dxright = (v1.X - v0.X) / (v1.Y - v0.Y) * yInc;
		dzright = (v1.Z - v0.Z) / (v1.Y - v0.Y) * yInc;
	}

	if (patches_Vertices[PatchNo][0].Y < -0.999 && abs(patches_Vertices[PatchNo][0].X) < 0.001)
		v0.Y=v0.Y;

	double ThetaX1 = -patches_Longitude[PatchNo];
	double ThetaY1 = -patches_Latitude[PatchNo];

	double CosX1 = cos(ThetaX1);
	double SinX1 = sin(ThetaX1);
	double CosY1 = cos(-ThetaY1);
	double SinY1 = sin(-ThetaY1);

	double SinY1SinX1 = SinY1*SinX1;
	double CosY1CosX1 = CosY1*CosX1;
	double SinY1CosX1 = SinY1*CosX1;
	double CosY1SinX1 = CosY1*SinX1;

	double ThetaX2 = m_Longitude;
	double ThetaY2 = m_Latitude;

	double CosX2 = cos(ThetaX2);
	double SinX2 = sin(ThetaX2);
	double CosY2 = cos(-ThetaY2);
	double SinY2 = sin(-ThetaY2);

	double SinY2SinX2 = SinY2*SinX2;
	double CosY2CosX2 = CosY2*CosX2;
	double SinY2CosX2 = SinY2*CosX2;
	double CosY2SinX2 = CosY2*SinX2;

	// If there are any scan lines in the upper section
    if (abs(v1.Y - v0.Y) >= abs(yInc))
    {
        // Set the scan lines from y0 to y1
        for (double y = v0.Y; y < v1.Y; y += yInc)
        {
			// Transform coords to the patch position
			// Rotate about x-axis for latitude, then y-axis for longitude
			double xleftTemp = xleft*CosX1 + y*SinY1SinX1 - zleft*CosY1SinX1;
			double yleftTemp = y*CosY1 + zleft*SinY1;
			double zleftTemp = xleft*SinX1 -y*SinY1CosX1 + zleft*CosY1CosX1;
			double xrightTemp = xright*CosX1 + y*SinY1SinX1 - zright*CosY1SinX1;
			double yrightTemp = y*CosY1 + zright*SinY1;
			double zrightTemp = xright*SinX1 -y*SinY1CosX1 + zright*CosY1CosX1;

			// Transform coords to the viewing position
			// Rotate about x-axis for latitude, then y-axis for longitude
			double xleftTrans = xleftTemp*CosX2 + yleftTemp*SinY2SinX2 - zleftTemp*CosY2SinX2;
			double yleftTrans = yleftTemp*CosY2 + zleftTemp*SinY2;
			double zleftTrans = xleftTemp*SinX2 - yleftTemp*SinY2CosX2 + zleftTemp*CosY2CosX2;
			double xrightTrans = xrightTemp*CosX2 + yrightTemp*SinY2SinX2 - zrightTemp*CosY2SinX2;
			double yrightTrans = yrightTemp*CosY2 + zrightTemp*SinY2;
			double zrightTrans = xrightTemp*SinX2 - yrightTemp*SinY2CosX2 + zrightTemp*CosY2CosX2;

			// Test whether line is behind viewing pane
			if (zleftTrans < 0 || zrightTrans < 0)
				return 0;

			SetTileLine(xleftTrans, xrightTrans, zleftTrans, zrightTrans, yleftTrans, yrightTrans);

            // Set the left/right limits for the next line
            xleft += dxleft;
            xright += dxright;
            zleft += dzleft;
            zright += dzright;
        }
		xleft = v0.X + (v1.Y - v0.Y) / yInc * dxleft;
		xright = v0.X + (v1.Y - v0.Y) / yInc * dxright;
		zleft = v0.Z + (v1.Y - v0.Y) / yInc * dzleft;
		zright = v0.Z + (v1.Y - v0.Y) / yInc * dzright;
    }
	else
	{
		xleft = v0.X; xright = v1.X;
		zleft = v0.Z; zright = v1.Z;
	}

	dxright = 0; 
	dzright = 0;

	// Initialise the increments between lines from y1 to y2
	if (v2.Y - v1.Y != 0)
	{
		dxright = (v2.X - v1.X) / (v2.Y - v1.Y) * yInc;
		dzright = (v2.Z - v1.Z) / (v2.Y - v1.Y) * yInc;
	}

    // Set the scan lines from y1 to y2
    for (double y = v1.Y; y <= v2.Y; y += yInc)
    {
		// Transform coords to the patch position
		// Rotate about x-axis for latitude, then y-axis for longitude
		double xleftTemp = xleft*CosX1 + y*SinY1SinX1 - zleft*CosY1SinX1;
		double yleftTemp = y*CosY1 + zleft*SinY1;
		double zleftTemp = xleft*SinX1 -y*SinY1CosX1 + zleft*CosY1CosX1;
		double xrightTemp = xright*CosX1 + y*SinY1SinX1 - zright*CosY1SinX1;
		double yrightTemp = y*CosY1 + zright*SinY1;
		double zrightTemp = xright*SinX1 -y*SinY1CosX1 + zright*CosY1CosX1;

		// Transform coords to the viewing position
		// Rotate about x-axis for latitude, then y-axis for longitude
		double xleftTrans = xleftTemp*CosX2 + yleftTemp*SinY2SinX2 - zleftTemp*CosY2SinX2;
		double yleftTrans = yleftTemp*CosY2 + zleftTemp*SinY2;
		double zleftTrans = xleftTemp*SinX2 - yleftTemp*SinY2CosX2 + zleftTemp*CosY2CosX2;
		double xrightTrans = xrightTemp*CosX2 + yrightTemp*SinY2SinX2 - zrightTemp*CosY2SinX2;
		double yrightTrans = yrightTemp*CosY2 + zrightTemp*SinY2;
		double zrightTrans = xrightTemp*SinX2 - yrightTemp*SinY2CosX2 + zrightTemp*CosY2CosX2;

		// Test whether line is behind viewing pane
		if (zleftTrans < 0 || zrightTrans < 0)
			return 0;

		SetTileLine(xleftTrans, xrightTrans, zleftTrans, zrightTrans, yleftTrans, yrightTrans);

        // Set the left/right limits for the next line
        xleft += dxleft;
        xright += dxright;
        zleft += dzleft;
        zright += dzright;
    }
	return 1;
}

bool LineInView(double XLeft, double XRight, double YLeft, double YRight)
{
    // Determine the pixel positions of this tile on the bitmap
    int iXLeft = (int)((XLeft - m_ImagePlane.Left) * m_ImageRatio);
    int iXRight = (int)((XRight - m_ImagePlane.Left) * m_ImageRatio);
    int iYLeft = (int)((YLeft - m_ImagePlane.Top) * m_ImageRatio);
    int iYRight = (int)((YRight - m_ImagePlane.Top) * m_ImageRatio);

    if (iYLeft < 0 && iYRight < 0 || iXLeft < 0 && iXRight < 0 ||
	    iYLeft > m_BitmapHeight && iYRight > m_BitmapHeight || 
		iXLeft > m_BitmapWidth && iXRight > m_BitmapWidth)
		return false;

	return true;
}

void SetTileLine(double xLeft, double xRight, double zLeft, double zRight, double yLeft, double yRight) 
{
	if (!LineInView(xLeft, xRight, yLeft, yRight))
	{
		MoveToNextLine();
		return;
	}

	BYTE r = 0, g = 0, b = 0;

    double XPos = xLeft;
    double YPos = yLeft;
	double deltaX = xRight - xLeft;
	double deltaY = yRight - yLeft;
	double deltaZ = zRight - zLeft;

	// Calculate the distance (in the xz plane) to scan along for this line
	double xyzdistance = sqrt(deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ);
    double xInc = 0, yInc = 0;

    if (xyzdistance > 0)
	{
		// Record the step sizes for x, y and z
		xInc = m_Resolution * deltaX / xyzdistance;
		yInc = m_Resolution * deltaY / xyzdistance;
	}

	// For each point along the line until a 'stop' pixel is reached
	while (patches_PixelData[m_TilePatchNo][m_TilePixelIndex].Red > 0 || 
		   patches_PixelData[m_TilePatchNo][m_TilePixelIndex].Green > 0 || 
		   patches_PixelData[m_TilePatchNo][m_TilePixelIndex].Blue > 0)
    {
		// Determine the pixel position on the bitmap
        int x = (int)((XPos - m_ImagePlane.Left) * m_ImageRatio + 0.5);
		int y = (int)((YPos - m_ImagePlane.Top) * m_ImageRatio + 0.5);

		// Set this pixel
		BYTE* row = m_Scan0 + y * m_Stride;
		SetPixel(x, y, row);

		// Set the adjacent x-pixel
		x++;
		SetPixel(x, y, row);

		// Set the adjacent y-pixel
		x--; y++;
		row = m_Scan0 + y * m_Stride;
		SetPixel(x, y, row);

		m_TilePixelIndex++;

        XPos += xInc;
        YPos += yInc;
    }
    // Move past the 'stop' pixel for this row
    m_TilePixelIndex++;
}

void SetPixel(int x, int y, BYTE* row)
{
	if (x < m_BitmapWidth && y < m_BitmapHeight && x >= 0 && y >= 0)
	{
		int iIndex = x*4;
		if (row[iIndex] == 0)
		{
			row[iIndex++] = patches_PixelData[m_TilePatchNo][m_TilePixelIndex].Blue;
			row[iIndex++] = patches_PixelData[m_TilePatchNo][m_TilePixelIndex].Green;
			row[iIndex++] = patches_PixelData[m_TilePatchNo][m_TilePixelIndex].Red;
		}
	}
}

void MoveToNextLine()
{
	// Move through each point along the line until a 'stop' pixel is reached
	while (patches_PixelData[m_TilePatchNo][m_TilePixelIndex].Red > 0 || 
		   patches_PixelData[m_TilePatchNo][m_TilePixelIndex].Green > 0 || 
		   patches_PixelData[m_TilePatchNo][m_TilePixelIndex].Blue > 0)
	{
		m_TilePixelIndex++;
	}
    // Move past the 'stop' pixel for this row
    m_TilePixelIndex++;
}
