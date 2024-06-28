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
        private BitmapData bitmapData = null;
        private clsSphere sphere = Model.Globals.Sphere;
        private float verticalView = 0;
        private float horizontalView = 0;
        private float maxVertical = 0;
        private float maxHorizontal = 0;
        private int bitmapWidth = 0;
        private int bitmapHeight = 0;

        public ImageDisplay()
        {
            verticalView = sphere.settings.VerticalView * Globals.DEG_TO_RAD / 2;
            horizontalView = sphere.settings.HorizontalView * Globals.DEG_TO_RAD / 2;
            float sphereResolution = sphere.settings.AngularResolution * Globals.DEG_TO_RAD / 2;

            maxHorizontal = (float)Math.Sin(horizontalView);
            maxVertical = (float)Math.Sin(verticalView);
            float stepSize = (float)Math.Sin(sphereResolution);

            // Initialise the bitmap
            bitmapWidth = (int)(maxHorizontal / stepSize);
            bitmapHeight = (int)(maxVertical / stepSize);

            // Bitmap resolution is higher than the image plane
            FinalBitmap = new Bitmap(bitmapWidth, bitmapHeight, PixelFormat.Format32bppRgb);
        }

        public void updateImage(float rayCountX, float rayCountY, Model.Globals.RGBQUAD colours)
        {
            // Get lat/long from rayCountX/Y
            float latitude = sphere.settings.LatitudeStart - rayCountY * sphere.settings.AngularResolution;
            float longitude = sphere.settings.LongitudeStart - rayCountX * sphere.settings.AngularResolution;

            latitude = latitude * Globals.DEG_TO_RAD;
            longitude = longitude * Globals.DEG_TO_RAD;

            // If latitude and longitude are within the field of view
            if (Math.Abs(latitude) <= verticalView
                && Math.Abs(longitude) <= horizontalView)
            {
                float verticalOffset = (float)Math.Sin(latitude);
                float horizontalOffset = (float)Math.Sin(longitude);

                int heightMidpoint = bitmapHeight / 2;
                int widthMidpoint = bitmapWidth / 2;

                // Determine the pixel position on the bitmap
                int x = (int)(widthMidpoint * (1 - horizontalOffset / maxHorizontal)) * 4;
                int y = (int)(heightMidpoint * (1 - verticalOffset / maxVertical));

                byte[] pixels = new byte[4];
                IntPtr ptrFirstPixel = bitmapData.Scan0;

                int currentLine = y * bitmapData.Stride;

                // Calculate new pixel value
                pixels[0] = colours.rgbBlue;
                pixels[1] = colours.rgbGreen;
                pixels[2] = colours.rgbRed;

                // Copy modified bytes back
                Marshal.Copy(pixels, 0, ptrFirstPixel + currentLine + x, 4);
            }
        }

        public Bitmap GetBitmap()
        {
            return FinalBitmap;
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
