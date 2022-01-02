using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace Model
{
    public unsafe class TracedRay
    {
        [DllImport("Unmanaged.dll")]
        static extern void HSVtoRGB(float h, float s, float v, byte* r, byte* g, byte* b);

        // The horizontal tilt of the surface at this point (-90 to 90 degrees)
        public List<float> xTiltValues { get; set; }
        // The vertical tilt of the surface at this point (-90 to 90 degrees)
        public List<float> yTiltValues { get; set; }
        // Overall RGB colour for this ray
        public Globals.RGBQUAD bmiColors;

        [StructLayout(LayoutKind.Sequential)]
        public struct RayDataType
        {
            public int[] ExternalPoints;
            public float[] ModulusValues;
            public float[] AngleValues;
            public double[] DistanceValues;
            public int BoundaryTotal;
        }

        public RayDataType RayData;

        public TracedRay(int[] externalPoints, float[] modulusValues, float[] angleValues, double[] distanceValues)
        {
            RayData.ExternalPoints = externalPoints;
            RayData.ModulusValues = modulusValues;
            RayData.AngleValues = angleValues;
            RayData.DistanceValues = distanceValues;
            RayData.BoundaryTotal = distanceValues == null ? 0 : distanceValues.Length;
        }

        public void SetColour()
        {
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
            float interiorExposure = sphere.InteriorExposure;
            float interiorSaturation = sphere.InteriorSaturation;

            try
            {
                // For each point on the ray
                for (int i = 0; i < RayData.ModulusValues.Length - 1; i++)
                {
                    if (RayData.DistanceValues[i] < startDistance)
                        continue;

                    if (Math.Abs(RayData.ModulusValues[i]) < 10)
                    {
                        if (activeIndex == 0 && IsSurfacePoint(i) && xTiltValues != null && yTiltValues != null)
                        {
                            ///// Set colour for surface point /////

                            if (Double.IsPositiveInfinity(RayData.DistanceValues[i])
                                || RayData.DistanceValues[i] > endDistance)
                                break;

                            float lightingAngleX = (sphere.LightingAngle - 90) * (float)Globals.DEG_TO_RAD;
                            float lightingAngleY = (160 - 90) * (float)Globals.DEG_TO_RAD;

                            // Modify the exposure value according to the XTilt, YTilt values using Lambert's Cosine Law
                            double xTilt = xTiltValues != null && xTiltValues.Count > 0 ? xTiltValues[i] : 0;
                            double yTilt = yTiltValues != null && yTiltValues.Count > 0 ? yTiltValues[i] : 0;
                            double tiltX = xTilt - lightingAngleX;
                            double tiltY = yTilt - lightingAngleY;
                            double xFactor = Math.Cos(tiltX);
                            double yFactor = Math.Cos(tiltY);

                            float surfaceContrast = sphere.SurfaceContrast / 10;

                            // Increase contrast of the exposure values
                            double exposureX = (xFactor - (float)0.5) * surfaceContrast + (float)0.5;
                            double exposureY = (yFactor - (float)0.5) * surfaceContrast + (float)0.5;

                            // Produce the final exposure value
                            float exposure = (float)(exposureX + exposureY) / 2;

                            exposure *= exposureValue / 10;

                            if (exposure < 0)
                                exposure = 0;
                            if (exposure > 1)
                                exposure = 1;

                            // Modify the exposure according to the position of the point between the start and end distances
                            //float range = (float)(endDistance - startDistance);
                            //float exposureFactor = (float)(RayData.DistanceValues[i] - startDistance) / range;

                            //if (exposureFactor > 1)
                            //{
                            //    exposureFactor = 1;
                            //}

                            // S & V set by exposure value
                            float Lightness = exposureValue;// *(1 - exposureFactor);
                            float Saturation = Lightness * saturation / 10;

                            IncreaseRGB(ref totalRGB, i, Saturation, Lightness);
                        }
                        else if (activeIndex == 1)
                        {
                            ///// Set colour for volume point /////

                            if (Double.IsPositiveInfinity(RayData.DistanceValues[i + 1])
                                || RayData.DistanceValues[i + 1] > endDistance)
                                break;

                            // Get distance between points
                            double distance = RayData.DistanceValues[i + 1] - RayData.DistanceValues[i];

                            // S & V set by distance * exposure value
                            float Lightness = (float)distance * exposureValue;
                            float Saturation = Lightness * saturation;

                            IncreaseRGB(ref totalRGB, i, Saturation, Lightness);
                        }
                    }
                }
            }
            catch (Exception e)
            {
                MessageBox.Show("Error tracing ray: " + e.Message, "Trace Ray Error");
            }

            // Colour the inside of the set if visible
            if (RayData.ModulusValues.Length == 2 && RayData.ModulusValues[0] < 2 && RayData.ModulusValues[1] == 0)
            {
                IncreaseRGB(ref totalRGB, 0, interiorExposure * interiorSaturation, interiorExposure);
            }

            bmiColors.rgbRed = (byte)totalRGB.rgbRed;
            bmiColors.rgbGreen = (byte)totalRGB.rgbGreen;
            bmiColors.rgbBlue = (byte)totalRGB.rgbBlue;
        }

        private void IncreaseRGB(ref Globals.RGBTRIPLE totalRGB, int i, float Saturation, float Lightness)
        {
            byte r, g, b;

            // Get Hue from the orbit angle
            float Hue = (float)(RayData.AngleValues[i] * 57.2957795 * 2);

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

        /// <summary>
        /// The number of sampled points along the ray
        /// </summary>
        public int Length
        {
            get { return RayData.ExternalPoints.Length; }
        }

        /// <summary>
        /// Return distance values for boundary positions.
        /// </summary>
        public double[] Boundaries
        {
            get { return RayData.DistanceValues; }
        }

        /// <summary>
        /// Return angle values for boundary positions.
        /// </summary>
        public float[] AngleValues
        {
            get { return RayData.AngleValues; }
        }

        /// <summary>
        /// Return modulus values for boundary positions.
        /// </summary>
        public float[] ModulusValues
        {
            get { return RayData.ModulusValues; }
        }

        /// <summary>
        /// Return external points for boundary positions.
        /// </summary>
        public int[] ExternalPoints
        {
            get { return RayData.ExternalPoints; }
        }

        /// <summary>
        /// Return the total number of boundary positions.
        /// </summary>
        public int BoundaryTotal
        {
            get { return RayData.DistanceValues.Length; }
        }

        /// <summary>
        /// Return distance value for the specified point.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double Boundary(int index)
        {
            return RayData.DistanceValues[index]; 
        }

        /// <summary>
        /// Return angle value for the specified point.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public float Angle(int index)
        {
            return RayData.AngleValues[index];
        }

        /// <summary>
        /// Return whether this point is on a surface boundary.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public bool IsSurfacePoint(int index)
        {
            if (index > 0 && index < BoundaryTotal && !double.IsPositiveInfinity(RayData.DistanceValues[index]))
            {
                return RayData.ExternalPoints[index - 1] == 1 && RayData.ExternalPoints[index] == 0;
            }
            return false;
        }
    }
}
