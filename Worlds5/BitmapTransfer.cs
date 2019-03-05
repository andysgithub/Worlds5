using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;
using System.Runtime.InteropServices;

namespace Worlds5
{
    public unsafe class BitmapTransfer
    {
        public static void Transfer(Bitmap FastBitmap, byte* Scan0, int Stride)
        {
            DataClasses.Globals.PixelData[,] TileData = RenderTile.Bitmap;

            if (RenderTile.Rectangle.Top < 0 && RenderTile.Rectangle.Bottom < 0 ||
                RenderTile.Rectangle.Left < 0 && RenderTile.Rectangle.Right < 0 ||
                RenderTile.Rectangle.Top > FastBitmap.Height && RenderTile.Rectangle.Bottom > FastBitmap.Height ||
                RenderTile.Rectangle.Left > FastBitmap.Width && RenderTile.Rectangle.Right > FastBitmap.Width)
                return;

            Point rectLoc = new Point(RenderTile.Rectangle.X, RenderTile.Rectangle.Y);
            Size rectSize = new Size(RenderTile.Rectangle.Width, RenderTile.Rectangle.Height);

            int xStart = 0;
            int yStart = 0;

            if (rectLoc.X < 0)
            {
                rectSize.Width = rectSize.Width + rectLoc.X;
                xStart = -rectLoc.X;
                rectLoc.X = 0;
            }
            if (rectLoc.Y < 0)
            {
                rectSize.Height = rectSize.Height + rectLoc.Y;
                yStart = -rectLoc.Y;
                rectLoc.Y = 0;
            }

            if (RenderTile.Rectangle.Bottom > FastBitmap.Height)
                rectSize.Height = rectSize.Height - (RenderTile.Rectangle.Bottom - FastBitmap.Height);
            if (RenderTile.Rectangle.Right > FastBitmap.Width)
                rectSize.Width = rectSize.Width - (RenderTile.Rectangle.Right - FastBitmap.Width);

            if (rectSize.Height == 0 || rectSize.Width == 0)
                return;

            RenderTile.Rectangle = new Rectangle(rectLoc, rectSize);

            try
            {
                // Transfer TileData to FastBitmap
                for (int y = 0; y < RenderTile.Rectangle.Height; y++)
                {
                    byte* row = Scan0 + ((y + rectLoc.Y) * Stride);
                    for (int x = 0; x < RenderTile.Rectangle.Width; x++)
                    {
                        if (TileData[x + xStart, y + yStart].red != 0 ||
                            TileData[x + xStart, y + yStart].green != 0 ||
                            TileData[x + xStart, y + yStart].blue != 0)
                        {
                            row[(x + rectLoc.X) * 4] = TileData[x + xStart, y + yStart].blue;
                            row[(x + rectLoc.X) * 4 + 1] = TileData[x + xStart, y + yStart].green;
                            row[(x + rectLoc.X) * 4 + 2] = TileData[x + xStart, y + yStart].red;
                        }
                    }
                }
            }
            catch
            { }
        }

    }
}
