using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

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
        public Globals.RGB_QUAD bmiColors;

        public struct RenderingParams
        {
            public int activeIndex;
            public float startDistance;
            public float endDistance;
            public float exposureValue;
            public float interiorExposure;
            public float saturation;
            public float interiorSaturation;
            public float lightingAngle;
            public float lightElevationAngle;
            public float surfaceContrast;
            public float colourCompression;
            public float colourOffset;

            public RenderingParams(clsSphere.Settings settings)
            {
                activeIndex = settings.ActiveIndex;
                startDistance = settings.SphereRadius;
                endDistance = settings.SphereRadius + settings.MaxSamples[activeIndex] * settings.SamplingInterval[activeIndex];
                exposureValue = settings.ExposureValue[activeIndex];
                interiorExposure = settings.ExposureValue[2];
                saturation = settings.Saturation[activeIndex];
                interiorSaturation = settings.Saturation[2];
                lightingAngle = settings.LightingAngle;
                lightElevationAngle = settings.LightElevationAngle;
                surfaceContrast = settings.SurfaceContrast;
                colourCompression = settings.ColourCompression;
                colourOffset = settings.ColourOffset;
            }
        };

        [StructLayout(LayoutKind.Sequential)]
        [Serializable]
        public struct RayDataType
        {
            public int[] ExternalPoints;
            public float[] ModulusValues;
            public float[] AngleValues;
            public float[] DistanceValues;
            public int BoundaryTotal;
        }

        private const int MAX_POINTS = 100;

        [StructLayout(LayoutKind.Sequential)]
        public struct RayDataTypeIntermediate
        {
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = MAX_POINTS)]
            public int[] ExternalPoints;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = MAX_POINTS)]
            public float[] ModulusValues;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = MAX_POINTS)]
            public float[] AngleValues;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = MAX_POINTS)]
            public float[] DistanceValues;
            public int BoundaryTotal;
            public int ArraySize;
        }

        public RayDataType RayData;
        private RenderingParams m_renderParams;

        public TracedRay(int[] externalPoints, float[] modulusValues, float[] angleValues, float[] distanceValues, RenderingParams renderParams)
        {
            RayData.ExternalPoints = externalPoints;
            RayData.ModulusValues = modulusValues;
            RayData.AngleValues = angleValues;
            RayData.DistanceValues = distanceValues;
            RayData.BoundaryTotal = distanceValues == null ? 0 : distanceValues.Length;
            m_renderParams = renderParams;
        }

        public void SetColour()
        {
            Globals.RGB_TRIPLE totalRGB;

            // Initialise RGB to 0,0,0
            totalRGB.rgbRed = 0;
            totalRGB.rgbGreen = 0;
            totalRGB.rgbBlue = 0;

            clsSphere sphere = Model.Globals.Sphere;

            int activeIndex = sphere.settings.ActiveIndex;
            float startDistance = m_renderParams.startDistance;
            float endDistance = m_renderParams.endDistance;
            float exposureValue = m_renderParams.exposureValue;
            float saturation = m_renderParams.saturation;
            float interiorExposure = m_renderParams.interiorExposure;
            float interiorSaturation = m_renderParams.interiorSaturation;

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

                            if (Single.IsPositiveInfinity(RayData.DistanceValues[i])
                                || RayData.DistanceValues[i] > endDistance)
                                break;

                            // Light position parameters
                            float lightingAngleXY = -sphere.settings.LightingAngle * (float)Globals.DEG_TO_RAD;
                            float lightElevationAngle = sphere.settings.LightElevationAngle * (float)Globals.DEG_TO_RAD;

                            // Calculate light direction
                            Vector3 lightDirection = new Vector3(
                                (float)(Math.Cos(lightingAngleXY) * Math.Cos(lightElevationAngle)),
                                (float)(Math.Sin(lightingAngleXY) * Math.Cos(lightElevationAngle)),
                                (float)Math.Sin(lightElevationAngle)
                            );
                            lightDirection = lightDirection.Normalize();

                            // Get tilt values
                            float xTilt = xTiltValues != null && xTiltValues.Count > 0 ? xTiltValues[i] : 0;
                            float yTilt = yTiltValues != null && yTiltValues.Count > 0 ? yTiltValues[i] : 0;

                            // Calculate the surface normal
                            Vector3 surfaceNormal = new Vector3(
                                (float)-Math.Sin(xTilt),
                                (float)-Math.Sin(yTilt),
                                (float)Math.Sqrt(1 - Math.Sin(xTilt) * Math.Sin(xTilt) - Math.Sin(yTilt) * Math.Sin(yTilt))
                            );
                            surfaceNormal = surfaceNormal.Normalize();

                            // Calculate the dot product
                            float dotProduct = Vector3.Dot(lightDirection, surfaceNormal);

                            // Ensure the dot product is in the range [0, 1]
                            float tiltValue = Math.Max(0, dotProduct);

                            float surfaceContrast = sphere.settings.SurfaceContrast / 10f;

                            // Apply contrast
                            float contrastValue = (tiltValue - 0.5f) * (1 + surfaceContrast) + 0.5f;
                            contrastValue = Math.Max(0, Math.Min(1, contrastValue));

                            // Calculate final lightness
                            float lightness = contrastValue * exposureValue / 10f;
                            lightness = Math.Max(0, Math.Min(1, lightness));

                            // Modify the exposure according to the position of the point between the start and end distances
                            //float range = (float)(endDistance - startDistance);
                            //float exposureFactor = (float)(RayData.DistanceValues[i] - startDistance) / range;
                            //exposureFactor = Math.Min(1, exposureFactor));

                            // S & V set by exposure value
                            float Lightness = lightness;// *(1 - exposureFactor);
                            float Saturation = Lightness * saturation / 10;

                            IncreaseRGB(ref totalRGB, i, Saturation, Lightness);
                        }
                        else if (activeIndex == 1)
                        {
                            ///// Set colour for volume point /////

                            if (Single.IsPositiveInfinity(RayData.DistanceValues[i + 1])
                                || RayData.DistanceValues[i + 1] > endDistance)
                                break;

                            // Get distance between points
                            float distance = RayData.DistanceValues[i + 1] - RayData.DistanceValues[i];

                            // S & V set by distance * exposure value
                            float Lightness = (float)distance * exposureValue / 10;
                            float Saturation = Lightness * saturation;

                            IncreaseRGB(ref totalRGB, i, Saturation, Lightness);
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Error tracing ray: {0} ", e.Message);
            }

            // Colour the inside of the set if visible
            if (RayData.ModulusValues.Length == 2 && RayData.ModulusValues[0] < 2 && RayData.ModulusValues[1] == 0)
            {
                float Lightness = interiorExposure / 10;
                float Saturation = Lightness * interiorSaturation / 10;
                IncreaseRGB(ref totalRGB, 0, Saturation, Lightness);
            }

            bmiColors.rgbRed = (byte)totalRGB.rgbRed;
            bmiColors.rgbGreen = (byte)totalRGB.rgbGreen;
            bmiColors.rgbBlue = (byte)totalRGB.rgbBlue;
        }

        private void IncreaseRGB(ref Globals.RGB_TRIPLE totalRGB, int i, float Saturation, float Lightness)
        {
            byte r, g, b;
            clsSphere sphere = Model.Globals.Sphere;

            float compression = sphere.settings.ColourCompression;
            float offset = sphere.settings.ColourOffset;

            // Get Hue from the orbit angle
            float Hue = (float)(RayData.AngleValues[i] * 57.2957795 * compression) + offset;

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
        public float[] Boundaries
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
            get {
                return RayData.DistanceValues == null ? 0 : RayData.DistanceValues.Length; 
            }
        }

        /// <summary>
        /// Return distance value for the specified point.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public float Boundary(int index)
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
            if (index > 0 && index < BoundaryTotal && !float.IsPositiveInfinity(RayData.DistanceValues[index]))
            {
                return RayData.ExternalPoints[index - 1] == 1 && RayData.ExternalPoints[index] == 0;
            }
            return false;
        }
    }
}
