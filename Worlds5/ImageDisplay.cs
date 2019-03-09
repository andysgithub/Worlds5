using System;
using System.Drawing;
using System.Drawing.Imaging;
using Model;

namespace Worlds5
{
    public unsafe class ImageDisplay
    {
        private Bitmap FinalBitmap = null;
        private clsSphere sphere = Model.Globals.Sphere;
        private double verticalView = 0;
        private double horizontalView = 0;
        private double maxVertical = 0;
        private double maxHorizontal = 0;

        public ImageDisplay()
        {
            verticalView = sphere.VerticalView * Globals.DEG_TO_RAD / 2;
            horizontalView = sphere.HorizontalView * Globals.DEG_TO_RAD / 2;
            double sphereResolution = sphere.AngularResolution * Globals.DEG_TO_RAD / 2;

            maxHorizontal = Math.Sin(horizontalView);
            maxVertical = Math.Sin(verticalView);
            double stepSize = Math.Sin(sphereResolution);

            // Initialise the bitmap
            int xPixels = (int)(maxHorizontal / stepSize);
            int yPixels = (int)(maxVertical / stepSize);

            // Bitmap resolution is higher than the image plane
            FinalBitmap = new Bitmap(xPixels, yPixels, PixelFormat.Format32bppRgb);
        }

        public void updateImage(double degreesLat, double degreesLong, Model.Globals.RGBQUAD colours)
        {
            double latitude = degreesLat * Globals.DEG_TO_RAD;
            double longitude = degreesLong * Globals.DEG_TO_RAD;

            // If latitude and longitude are within the field of view
            if (Math.Abs(latitude) <= verticalView
                && Math.Abs(longitude) <= horizontalView)
            {
                BitmapData bmd = LockBitmap(ref FinalBitmap);

                double verticalOffset = Math.Sin(latitude);
                double horizontalOffset = Math.Sin(longitude);

                int heightMidpoint = bmd.Height / 2;
                int widthMidpoint = bmd.Width / 2;

                // Determine the pixel position on the bitmap
                int x = (int)(widthMidpoint * (1 - horizontalOffset / maxHorizontal));
                int y = (int)(heightMidpoint * (1 - verticalOffset / maxVertical));

                if (x >= 0 && y >= 0 && x < bmd.Width && y < bmd.Height)
                {
                    // Plot this colour on the bitmap
                    byte* row = (byte*)(bmd.Scan0 + y * bmd.Stride);
                    SetPixel(x, row, colours);
                }
                FinalBitmap.UnlockBits(bmd);
            }
        }

        public Bitmap getBitmap()
        {
            return FinalBitmap;
        }

        private void SetPixel(int x, byte* row, Model.Globals.RGBQUAD colours)
        {
            int iIndex = x * 4;

            try
            {
                //if (row[iIndex] == 0)
                {
                    row[iIndex++] = colours.rgbBlue;
                    row[iIndex++] = colours.rgbGreen;
                    row[iIndex++] = colours.rgbRed;
                }
            }
            catch
            {
            }
        }

        private BitmapData LockBitmap(ref Bitmap FinalBitmap)
        {
            return FinalBitmap.LockBits(
                            new Rectangle(0, 0, FinalBitmap.Width, FinalBitmap.Height),
                            ImageLockMode.ReadOnly, FinalBitmap.PixelFormat);
        }
    }
}
