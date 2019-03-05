using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Model
{
    public unsafe class SurfacePoint
    {
        [DllImport("Unmanaged.dll")]
        static extern void HSVtoRGB(float h, float s, float v, byte* r, byte* g, byte* b);

        private float modulus; 
        private float angle;
        private double distance;

        private float xTilt;
        private float yTilt;

        // RGB colour
        public Globals.RGBQUAD bmiColors;

        public SurfacePoint(float modulus, float angle, double distance)
        {
            this.modulus = modulus;
            this.angle = angle;
            this.distance = distance;
        }

        public void SetColour(float exposureValue, float saturation, double startDistance, double endDistance)
        {
            byte r = 0, g = 0, b = 0;

            if (Math.Abs(modulus) < 10)
            {
                // Get Hue from the orbit angle
                float Hue = (float)(angle * 57.2957795 * 2);

                // Modify the exposure according to the position of the point between the start and end distances
                float range = (float)(endDistance - startDistance);
                float exposureFactor = (float)(distance - startDistance) / range;

                if (exposureFactor > 1)
                {
                    exposureFactor = 1;
                }

                float Lightness = exposureValue * (1 - exposureFactor);
                float Saturation = Lightness * saturation / 10;

                // Limit S & V to 1 maximum
                Saturation = Saturation > 1 ? 1 : Saturation;
                Lightness = Lightness > 1 ? 1 : Lightness;

                // Convert HSV to RGB
                HSVtoRGB(Hue, Saturation, Lightness, &r, &g, &b);
            }

            bmiColors.rgbRed = r;
            bmiColors.rgbGreen = g;
            bmiColors.rgbBlue = b;
        }

        /// <summary>
        /// The modulus value at the surface point
        /// </summary>
        public float Modulus
        {
            get { return modulus; }
        }

        /// <summary>
        /// The angle value at the surface point
        /// </summary>
        public float Angle
        {
            get { return angle; }
        }

        /// <summary>
        /// The distance value at the surface point
        /// </summary>
        public double Distance
        {
            get { return distance; }
        }

        /// <summary>
        /// The horizontal tilt of the surface at this point (-90 to 90 degrees)
        /// </summary>
        public float XTilt
        {
            get { return xTilt; }
            set { xTilt = value; }
        }

        /// <summary>
        /// The vertical tilt of the surface at this point (-90 to 90 degrees)
        /// </summary>
        public float YTilt
        {
            get { return yTilt; }
            set { yTilt = value; }
        }
    }
}
