using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;

namespace Model
{
    public unsafe class TracedRay
    {
        [DllImport("Unmanaged.dll")]
        static extern void HSVtoRGB(float h, float s, float v, byte* r, byte* g, byte* b);

        private List<bool> externalPoints;
        private List<float> modulusValues;
        private List<float> angleValues;
        private List<double> distanceValues;
        private List<float> xTiltValues;
        private List<float> yTiltValues;

        public int sampleCount;
        // Overall RGB colour for this ray
        public Globals.RGBQUAD bmiColors;

        public TracedRay(bool[] externalPoints, float[] modulusValues, float[] angleValues, double[] distanceValues)
        {
            this.externalPoints = new List<bool>(externalPoints);
            this.modulusValues = new List<float>(modulusValues);
            this.angleValues = new List<float>(angleValues);
            this.distanceValues = new List<double>(distanceValues);
            this.sampleCount = externalPoints.Length;
        }

        public void SetColour()
        {
            byte r, g, b;
            Globals.RGBTRIPLE totalRGB;

            // Initialise RGB to 0,0,0
            totalRGB.rgbRed = 0;
            totalRGB.rgbGreen = 0;
            totalRGB.rgbBlue = 0;

            clsSphere sphere = Model.Globals.Sphere;

            int activeIndex = sphere.ActiveIndex;

            double startDistance = sphere.StartDistance[activeIndex];
            double endDistance = sphere.EndDistance[activeIndex];
            float exposureValue = sphere.ExposureValue[activeIndex];
            float saturation = sphere.Saturation[activeIndex];

            // For each point on the ray
            for (int i = 0; i < modulusValues.Count - 1; i++)
            {
                if (distanceValues[i] < startDistance)
                    continue;

                if (Math.Abs(modulusValues[i]) < 10)
                {
                    if (activeIndex == 0 && isSurfacePoint(i) && xTiltValues != null && yTiltValues != null)
                    {
                        ///// Set colour for surface point /////

                        if (Double.IsPositiveInfinity(distanceValues[i])
                            || distanceValues[i] > endDistance)
                            break;

                        float lightingAngle = (sphere.LightingAngle + 90) * (float)Globals.DEG_TO_RAD;

                        // Modify the exposure value according to the XTilt, YTilt values using Lambert's Cosine Law
                        double xTilt = xTiltValues != null && xTiltValues.Count > 0 ? xTiltValues[i] : 0;
                        double yTilt = yTiltValues != null && yTiltValues.Count > 0 ? yTiltValues[i] : 0;
                        double tiltX = xTilt + lightingAngle;
                        double tiltY = yTilt + lightingAngle;
                        exposureValue = (float)(Math.Cos(tiltX) * Math.Cos(tiltY));

                        float surfaceContrast = sphere.SurfaceContrast / 10;

                        //// Increase contrast of the exposure value
                        exposureValue = (exposureValue * surfaceContrast * 2) - surfaceContrast;

                        exposureValue *= sphere.ExposureValue[0];

                        if (exposureValue < 0) exposureValue = 0;
                        if (exposureValue > 1) exposureValue = 1;

                        // Get Hue from the orbit angle
                        float Hue = (float)(angleValues[i] * 57.2957795 * 2);

                        // Modify the exposure according to the position of the point between the start and end distances
                        //float range = (float)(endDistance - startDistance);
                        //float exposureFactor = (float)(distanceValues[i] - startDistance) / range;

                        //if (exposureFactor > 1)
                        //{
                        //    exposureFactor = 1;
                        //}

                        float Lightness = exposureValue;// *(1 - exposureFactor);
                        float Saturation = Lightness * saturation / 10;

                        // Limit S & V to 1 maximum
                        Saturation = Saturation > 1 ? 1 : Saturation;
                        Lightness = Lightness > 1 ? 1 : Lightness;

                        // Convert HSV to RGB
                        HSVtoRGB(Hue, Saturation, Lightness, &r, &g, &b);

                        // Add result to RGB total
                        totalRGB.rgbRed += r;
                        totalRGB.rgbGreen += g;
                        totalRGB.rgbBlue += b;
                        // Limit RGB values to 255
                        totalRGB.rgbRed = totalRGB.rgbRed > 255 ? 255 : totalRGB.rgbRed;
                        totalRGB.rgbGreen = totalRGB.rgbGreen > 255 ? 255 : totalRGB.rgbGreen;
                        totalRGB.rgbBlue = totalRGB.rgbBlue > 255 ? 255 : totalRGB.rgbBlue;
                    }
                    else if (activeIndex == 1)
                    {
                        ///// Set colour for external point /////

                        if (Double.IsPositiveInfinity(distanceValues[i + 1])
                            || distanceValues[i + 1] > endDistance)
                            break;

                        // Get distance between points
                        double distance = distanceValues[i + 1] - distanceValues[i];

                        // Get Hue from the orbit angle
                        float Hue = (float)(angleValues[i] * 57.2957795 * 2);
                        // S & V set by distance * exposure value
                        float Lightness = (float)distance * exposureValue;
                        float Saturation = Lightness * saturation;

                        // Limit S & V to 1 maximum
                        Saturation = Saturation > 1 ? 1 : Saturation;
                        Lightness = Lightness > 1 ? 1 : Lightness;

                        // Convert HSV to RGB
                        HSVtoRGB(Hue, Saturation, Lightness, &r, &g, &b);
                        // Add result to RGB total
                        totalRGB.rgbRed += r;
                        totalRGB.rgbGreen += g;
                        totalRGB.rgbBlue += b;
                        // Limit RGB values to 255
                        totalRGB.rgbRed = totalRGB.rgbRed > 255 ? 255 : totalRGB.rgbRed;
                        totalRGB.rgbGreen = totalRGB.rgbGreen > 255 ? 255 : totalRGB.rgbGreen;
                        totalRGB.rgbBlue = totalRGB.rgbBlue > 255 ? 255 : totalRGB.rgbBlue;
                    }
                }
            }

            if (totalRGB.rgbGreen > 0)
            {
                ;
            }

            bmiColors.rgbRed = (byte)totalRGB.rgbRed;
            bmiColors.rgbGreen = (byte)totalRGB.rgbGreen;
            bmiColors.rgbBlue = (byte)totalRGB.rgbBlue;
        }

        /// <summary>
        /// The number of sampled points along the ray
        /// </summary>
        public int Length
        {
            get { return sampleCount; }
        }

        /// <summary>
        /// Return distance values for boundary positions.
        /// </summary>
        public List<double> Boundaries
        {
            get { return distanceValues; }
        }

        /// <summary>
        /// Return angle values for boundary positions.
        /// </summary>
        public List<float> AngleValues
        {
            get { return angleValues; }
        }

        /// <summary>
        /// Return modulus values for boundary positions.
        /// </summary>
        public List<float> ModulusValues
        {
            get { return modulusValues; }
        }

        /// <summary>
        /// Return external points for boundary positions.
        /// </summary>
        public List<bool> ExternalPoints
        {
            get { return externalPoints; }
        }

        /// <summary>
        /// The horizontal tilt of the surface at this point (-90 to 90 degrees)
        /// </summary>
        public List<float> XTiltValues
        {
            get { return xTiltValues; }
            set { xTiltValues = value; }
        }

        /// <summary>
        /// The vertical tilt of the surface at this point (-90 to 90 degrees)
        /// </summary>
        public List<float> YTiltValues
        {
            get { return yTiltValues; }
            set { yTiltValues = value; }
        }

        /// <summary>
        /// Return the total number of boundary positions.
        /// </summary>
        public int BoundaryTotal
        {
            get { return distanceValues.Count; }
        }

        /// <summary>
        /// Return distance value for the specified point.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double Boundary(int index)
        {
            return distanceValues[index]; 
        }

        /// <summary>
        /// Return angle value for the specified point.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public float Angle(int index)
        {
            return angleValues[index];
        }

        /// <summary>
        /// Return whether this point is on a surface boundary.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public bool isSurfacePoint(int index)
        {
            if (index > 0 && index < BoundaryTotal && !double.IsPositiveInfinity(distanceValues[index]))
            {
                return externalPoints[index - 1] && !externalPoints[index];
            }
            return false;
        }
    }
}
