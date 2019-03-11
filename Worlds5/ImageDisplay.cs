using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using Model;

namespace Worlds5
{
    public unsafe class ImageDisplay
    {
        private Bitmap FinalBitmap = null;
        private BitmapData bitmapData;
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
                //LockBitmap();

                double verticalOffset = Math.Sin(latitude);
                double horizontalOffset = Math.Sin(longitude);

                int heightMidpoint = FinalBitmap.Height / 2;
                int widthMidpoint = FinalBitmap.Width / 2;

                // Determine the pixel position on the bitmap
                int x = (int)(widthMidpoint * (1 - horizontalOffset / maxHorizontal));
                int y = (int)(heightMidpoint * (1 - verticalOffset / maxVertical));

                //if (x >= 0 && y >= 0 && x < bitmapData.Width && y < bitmapData.Height)
                //{
                //    // plot this colour on the bitmap
                //    byte* row = (byte*)(bitmapData.Scan0 + y * bitmapData.Stride);
                //    // todo: use direct access instead of setpixel
                //    SetPixel(x, row, colours);
                //}

                int bytesPerPixel = Bitmap.GetPixelFormatSize(FinalBitmap.PixelFormat) / 8;
                int byteCount = bitmapData.Stride * FinalBitmap.Height;
                byte[] pixels = new byte[byteCount];
                IntPtr ptrFirstPixel = bitmapData.Scan0;
                Marshal.Copy(ptrFirstPixel, pixels, 0, pixels.Length);

                int currentLine = y * bitmapData.Stride;

                // calculate new pixel value
                pixels[currentLine + x] = colours.rgbBlue;
                pixels[currentLine + x + 1] = colours.rgbGreen;
                pixels[currentLine + x + 2] = colours.rgbRed;

                // copy modified bytes back
                Marshal.Copy(pixels, 0, ptrFirstPixel, pixels.Length);
                //UnlockBitmap();
            }
        }

        public Bitmap GetBitmap()
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

        public void LockBitmap()
        {
            bitmapData = FinalBitmap.LockBits(
                            new Rectangle(0, 0, FinalBitmap.Width, FinalBitmap.Height),
                            ImageLockMode.ReadOnly, FinalBitmap.PixelFormat);
        }

        public void UnlockBitmap()
        {
            FinalBitmap.UnlockBits(bitmapData);
        }
    }
}
