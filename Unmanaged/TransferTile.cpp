#include "stdafx.h"
#include <math.h>
#include "unmanaged.h"
#include "declares.h"

EXPORT void __stdcall TransferTile(int BitmapWidth, int BitmapHeight, RectangleI Rect, pixel* TileData, BYTE* Scan0, int Stride)
{
    if ((Rect.Top < 0) && (Rect.Top + Rect.Height < 0) ||
		(Rect.Left < 0) && (Rect.Left + Rect.Width < 0) ||
		(Rect.Top > BitmapHeight) && (Rect.Top + Rect.Height > BitmapHeight) ||
		(Rect.Left > BitmapWidth) && (Rect.Left + Rect.Width > BitmapWidth))
		return;

    int rectLocX = Rect.Left;
    int rectLocY = Rect.Top;
    int rectWidth = Rect.Width;
	int rectHeight = Rect.Height;

    int xStart = 0;
    int yStart = 0;

    if (rectLocX < 0)
    {
        rectWidth = rectWidth + rectLocX;
        xStart = -rectLocX;
        rectLocX = 0;
    }
    if (rectLocY < 0)
    {
        rectHeight = rectHeight + rectLocY;
        yStart = -rectLocY;
        rectLocY = 0;
    }

    if (Rect.Top + Rect.Height > BitmapHeight)
        rectHeight = rectHeight - (Rect.Top + Rect.Height - BitmapHeight);
    if (Rect.Left + Rect.Width > BitmapWidth)
        rectWidth = rectWidth - (Rect.Left + Rect.Width - BitmapWidth);

    if (rectHeight == 0 || rectWidth == 0)
        return;

    // Transfer TileData to FastBitmap
    for (int y = 0; y < rectHeight; y++)
    {
        BYTE* row = Scan0 + (y * Stride);
        for (int x = 0; x < rectWidth; x++)
        {
			int Index = (x + xStart) * rectWidth + (y + yStart);
            if (TileData[Index].Red != 0 ||
                TileData[Index].Green != 0 ||
                TileData[Index].Blue != 0)
            {
                row[x * 4] = TileData[Index].Blue;
                row[x * 4 + 1] = TileData[Index].Green;
                row[x * 4 + 2] = TileData[Index].Red;
            }
        }
    }
}